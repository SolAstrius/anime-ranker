import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildNormalityLine, malScoreSummary } from "./lib/analysis";
import { createEloState, recordOutcome, selectPair } from "./lib/elo";
import { fetchAnimeMedia } from "./lib/jikan";
import { parseMalExport } from "./lib/mal";
import { createRng } from "./lib/random";
import {
  buildResults,
  resultsToCsv,
  resultsToJson,
  BlendingParams,
  computeEloWeight,
} from "./lib/results";
import {
  fitNormal,
  normalCdf,
  percentile,
  score10FromPercentile,
} from "./lib/scoring";
import {
  AnimeEntry,
  AnimeMedia,
  AnimeResult,
  EloState,
  MALExport,
  Rating,
} from "./lib/types";

const STORAGE_KEY = "anime-ranker-state-v1";

const comparisonTargets = (n: number, comparisonScale: number) => {
  if (n <= 0 || comparisonScale <= 0) {
    return { meaningfulMin: 0, optimal: 0, excessive: 0 };
  }
  const total = (avgGames: number) =>
    Math.max(1, Math.round(((avgGames * n) / 2) * comparisonScale));
  return {
    meaningfulMin: total(2),
    optimal: total(4),
    excessive: total(7),
  };
};

const safeBaseName = (name: string) => {
  const cleaned = name
    .split("")
    .filter((ch) => /[a-zA-Z0-9\-_. ]/.test(ch))
    .join("")
    .trim();
  return cleaned || "results";
};

const downloadBlob = (name: string, data: string, type: string) => {
  const blob = new Blob([data], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
};

type AppliedSettings = {
  comparisonsTarget: number;
  kFactor: number;
  avoidRepeats: boolean;
  assumeDroppedLower: boolean;
  statuses: string[];
  malNormality: number;
  normalityMultiplier: number;
};

type StoredEloState = {
  ratings: [number, Rating][];
  comparisons: number;
  skips: number;
  pairHistory: string[];
  kFactor: number;
  initialRating: number;
};

type StoredState = {
  version: 1;
  exportData: MALExport;
  fileName: string | null;
  statusFilter: string[];
  showPosters: boolean;
  showEnglish: boolean;
  assumeDroppedLower: boolean;
  avoidRepeats: boolean;
  kFactor: number;
  comparisonsTarget: number;
  settingsConfirmed: boolean;
  appliedSettings: AppliedSettings | null;
  appliedAnimeIds: number[];
  scoreMeanTarget: number | null;
  autoDecisions: number;
  currentPair: [number, number] | null;
  eloState: StoredEloState | null;
};

const serializeEloState = (elo: EloState): StoredEloState => ({
  ratings: Array.from(elo.ratings.entries()),
  comparisons: elo.comparisons,
  skips: elo.skips,
  pairHistory: Array.from(elo.pairHistory),
  kFactor: elo.kFactor,
  initialRating: elo.initialRating,
});

const deserializeEloState = (raw: StoredEloState): EloState => ({
  ratings: new Map(raw.ratings),
  comparisons: raw.comparisons,
  skips: raw.skips,
  pairHistory: new Set(raw.pairHistory),
  kFactor: raw.kFactor,
  initialRating: raw.initialRating,
});

type SkipDecision = {
  outcome: number;
  reason: string;
};

const NormalityChart = ({
  scores,
  mu,
  sigma,
}: {
  scores: number[];
  mu: number;
  sigma: number;
}) => {
  const line = useMemo(
    () => buildNormalityLine(scores, { mu, sigma }),
    [scores, mu, sigma],
  );
  const width = 280;
  const height = 110;
  const paddingX = 10;
  const paddingY = 12;

  const scaleX = (value: number) =>
    paddingX + ((value - 1) / 9) * (width - paddingX * 2);
  const scaleY = (value: number) =>
    height - paddingY - value * (height - paddingY * 2);

  const userPath = line
    .map(
      (point, idx) =>
        `${idx === 0 ? "M" : "L"}${scaleX(point.x)},${scaleY(point.user)}`,
    )
    .join(" ");
  const idealPath = line
    .map(
      (point, idx) =>
        `${idx === 0 ? "M" : "L"}${scaleX(point.x)},${scaleY(point.ideal)}`,
    )
    .join(" ");

  const meanX = scaleX(mu);
  const meanDisplay = mu.toFixed(2);

  return (
    <svg
      className="normality-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Normality chart"
    >
      <path className="normality-ideal" d={idealPath} />
      <path className="normality-user" d={userPath} />
      <line
        className="normality-mean"
        x1={meanX}
        y1={paddingY}
        x2={meanX}
        y2={height - paddingY}
      />
      <text className="normality-mean-label" x={meanX} y={paddingY - 2}>
        {meanDisplay}
      </text>
      <line
        className="normality-axis"
        x1={paddingX}
        y1={height - paddingY}
        x2={width - paddingX}
        y2={height - paddingY}
      />
      <text className="normality-label" x={paddingX} y={height - 2}>
        1
      </text>
      <text className="normality-label" x={width / 2} y={height - 2}>
        5
      </text>
      <text className="normality-label" x={width - paddingX} y={height - 2}>
        10
      </text>
    </svg>
  );
};

const App = () => {
  const [hydrated, setHydrated] = useState(false);
  const [exportData, setExportData] = useState<MALExport | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string[]>([]);
  const [showPosters, setShowPosters] = useState(true);
  const [showEnglish, setShowEnglish] = useState(true);
  const [assumeDroppedLower, setAssumeDroppedLower] = useState(true);
  const [avoidRepeats, setAvoidRepeats] = useState(true);
  const [kFactor, setKFactor] = useState(32);
  const [comparisonsTarget, setComparisonsTarget] = useState(0);
  const [settingsConfirmed, setSettingsConfirmed] = useState(false);
  const [appliedSettings, setAppliedSettings] =
    useState<AppliedSettings | null>(null);
  const [appliedAnimeIds, setAppliedAnimeIds] = useState<number[]>([]);
  const [scoreMeanTarget, setScoreMeanTarget] = useState<number | null>(null);
  const [eloState, setEloState] = useState<EloState | null>(null);
  const [currentPair, setCurrentPair] = useState<[number, number] | null>(null);
  const [autoDecisions, setAutoDecisions] = useState(0);
  const [pairError, setPairError] = useState<string | null>(null);
  const [mediaCache, setMediaCache] = useState<Map<number, AnimeMedia | null>>(
    () => new Map(),
  );

  const rngRef = useRef(createRng(Date.now()));

  useEffect(() => {
    if (typeof window === "undefined") {
      setHydrated(true);
      return;
    }
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      setHydrated(true);
      return;
    }
    try {
      const parsed = JSON.parse(raw) as StoredState;
      if (!parsed || parsed.version !== 1) {
        setHydrated(true);
        return;
      }
      setExportData(parsed.exportData ?? null);
      setFileName(parsed.fileName ?? null);
      setStatusFilter(parsed.statusFilter ?? []);
      setShowPosters(parsed.showPosters ?? true);
      setShowEnglish(parsed.showEnglish ?? true);
      setAssumeDroppedLower(parsed.assumeDroppedLower ?? true);
      setAvoidRepeats(parsed.avoidRepeats ?? true);
      setKFactor(parsed.kFactor ?? 32);
      setComparisonsTarget(parsed.comparisonsTarget ?? 0);
      setSettingsConfirmed(parsed.settingsConfirmed ?? false);
      setAppliedSettings(parsed.appliedSettings ?? null);
      setAppliedAnimeIds(parsed.appliedAnimeIds ?? []);
      setScoreMeanTarget(parsed.scoreMeanTarget ?? null);
      setAutoDecisions(parsed.autoDecisions ?? 0);
      setCurrentPair(parsed.currentPair ?? null);
      setEloState(
        parsed.eloState ? deserializeEloState(parsed.eloState) : null,
      );
    } catch (err) {
      window.localStorage.removeItem(STORAGE_KEY);
    } finally {
      setHydrated(true);
    }
  }, []);

  const animeById = useMemo(() => {
    const map = new Map<number, AnimeEntry>();
    exportData?.anime.forEach((entry) => map.set(entry.animeId, entry));
    return map;
  }, [exportData]);

  const statusById = useMemo(() => {
    const map = new Map<number, string | null>();
    exportData?.anime.forEach((entry) => map.set(entry.animeId, entry.status));
    return map;
  }, [exportData]);

  const allStatuses = useMemo(() => {
    const statuses = new Set<string>();
    exportData?.anime.forEach((entry) => {
      if (entry.status) {
        statuses.add(entry.status);
      }
    });
    return Array.from(statuses).sort();
  }, [exportData]);

  const selectedEntries = useMemo(() => {
    if (!exportData) {
      return [];
    }
    const allowed = new Set(statusFilter);
    return exportData.anime.filter(
      (entry) => entry.status && allowed.has(entry.status),
    );
  }, [exportData, statusFilter]);

  const selectedIds = useMemo(() => {
    return selectedEntries.map((entry) => entry.animeId);
  }, [selectedEntries]);

  const completedCount = useMemo(() => {
    return selectedEntries.filter((entry) => entry.status === "Completed")
      .length;
  }, [selectedEntries]);

  const droppedCount = useMemo(() => {
    return selectedEntries.filter((entry) => entry.status === "Dropped").length;
  }, [selectedEntries]);

  const totalPairs = (selectedIds.length * (selectedIds.length - 1)) / 2;
  const pairScale = useMemo(() => {
    if (totalPairs <= 0) {
      return 1;
    }
    if (!assumeDroppedLower || !completedCount || !droppedCount) {
      return 1;
    }
    const allowedPairs = totalPairs - completedCount * droppedCount;
    return allowedPairs / totalPairs;
  }, [assumeDroppedLower, completedCount, droppedCount, totalPairs]);

  const malSummary = useMemo(() => {
    return malScoreSummary(selectedEntries);
  }, [selectedEntries]);

  const normalityMultiplier = useMemo(() => {
    const value = 1.2 - 0.9 * malSummary.normality;
    return Math.max(0.35, Math.min(1.2, value));
  }, [malSummary.normality]);

  const comparisonScale = pairScale * normalityMultiplier;
  const guidance = useMemo(
    () => comparisonTargets(selectedIds.length, comparisonScale),
    [selectedIds.length, comparisonScale],
  );

  const sliderMax = guidance.excessive;
  const clampPct = (value: number) => {
    const denom = sliderMax || 1;
    return Math.max(0, Math.min(100, (value / denom) * 100));
  };
  const rangeMarkerStyle = (value: number) => {
    const pct = clampPct(value);
    let translate = "-50%";
    if (pct < 8) {
      translate = "0%";
    } else if (pct > 92) {
      translate = "-100%";
    }
    return { left: `${pct}%`, transform: `translateX(${translate})` };
  };
  const rangeMarks = useMemo(
    () => [
      { label: "min", value: guidance.meaningfulMin },
      { label: "optimal", value: guidance.optimal },
      { label: "excessive", value: guidance.excessive },
    ],
    [guidance.meaningfulMin, guidance.optimal, guidance.excessive],
  );
  const currentValueStyle = rangeMarkerStyle(comparisonsTarget);

  useEffect(() => {
    if (!exportData || settingsConfirmed) {
      return;
    }
    const next = Math.max(0, guidance.optimal);
    setComparisonsTarget(Math.min(next, sliderMax));
  }, [exportData, settingsConfirmed, guidance.optimal, sliderMax]);

  useEffect(() => {
    if (!exportData) {
      return;
    }
    setStatusFilter((prev) => {
      const filtered = prev.filter((status) => allStatuses.includes(status));
      if (filtered.length) {
        return filtered;
      }
      const defaults = allStatuses.filter(
        (status) => status !== "Plan to Watch" && status !== "Watching",
      );
      return defaults.length ? defaults : allStatuses;
    });
  }, [allStatuses, exportData]);

  const ensureMedia = useCallback(
    async (animeId: number) => {
      if (mediaCache.has(animeId)) {
        return;
      }
      const media = await fetchAnimeMedia(animeId);
      setMediaCache((prev) => {
        const next = new Map(prev);
        next.set(animeId, media);
        return next;
      });
    },
    [mediaCache],
  );

  useEffect(() => {
    if (!currentPair || (!showPosters && !showEnglish)) {
      return;
    }
    currentPair.forEach((animeId) => {
      void ensureMedia(animeId);
    });
  }, [currentPair, showPosters, showEnglish, ensureMedia]);

  const resetRankingState = useCallback(() => {
    setSettingsConfirmed(false);
    setAppliedSettings(null);
    setAppliedAnimeIds([]);
    setScoreMeanTarget(null);
    setEloState(null);
    setCurrentPair(null);
    setAutoDecisions(0);
    setPairError(null);
  }, []);

  const handleFile = useCallback(
    async (file: File) => {
      const buffer = await file.arrayBuffer();
      const exportResult = parseMalExport(new Uint8Array(buffer), file.name);
      setExportData(exportResult);
      setFileName(file.name);
      setMediaCache(new Map());
      rngRef.current = createRng(Date.now());
      resetRankingState();
    },
    [resetRankingState],
  );

  const startRanking = useCallback(() => {
    if (!selectedIds.length) {
      return;
    }
    const newElo = createEloState(selectedIds, kFactor);
    setEloState(newElo);
    setCurrentPair(null);
    setAutoDecisions(0);
    setPairError(null);
    setSettingsConfirmed(true);
    setAppliedAnimeIds([...selectedIds]);
    setAppliedSettings({
      comparisonsTarget,
      kFactor,
      avoidRepeats,
      assumeDroppedLower,
      statuses: [...statusFilter],
      malNormality: malSummary.normality,
      normalityMultiplier,
    });
  }, [
    selectedIds,
    kFactor,
    comparisonsTarget,
    avoidRepeats,
    assumeDroppedLower,
    statusFilter,
    malSummary.normality,
    normalityMultiplier,
  ]);

  const appliedAssumeDroppedLower =
    appliedSettings?.assumeDroppedLower ?? assumeDroppedLower;
  const appliedAvoidRepeats = appliedSettings?.avoidRepeats ?? avoidRepeats;
  const appliedKFactor = appliedSettings?.kFactor ?? kFactor;
  const appliedStatuses = appliedSettings?.statuses ?? statusFilter;
  const appliedMalNormality =
    appliedSettings?.malNormality ?? malSummary.normality;
  const appliedNormalityMultiplier =
    appliedSettings?.normalityMultiplier ?? normalityMultiplier;

  const comparisonGoal =
    appliedSettings?.comparisonsTarget ?? comparisonsTarget;
  const done = Boolean(eloState && eloState.comparisons >= comparisonGoal);
  const inCombat = Boolean(settingsConfirmed && eloState && !done);
  const progressValue =
    !eloState || comparisonGoal <= 0
      ? 1
      : Math.min(1, eloState.comparisons / comparisonGoal);

  const activeAnimeIds = useMemo(
    () => (appliedAnimeIds.length ? appliedAnimeIds : selectedIds),
    [appliedAnimeIds, selectedIds],
  );

  const activeEntries = useMemo(() => {
    return activeAnimeIds
      .map((id) => animeById.get(id))
      .filter((entry): entry is AnimeEntry => Boolean(entry));
  }, [activeAnimeIds, animeById]);

  const malScoreMean = useMemo(() => {
    const scores = activeEntries
      .map((entry) => entry.myScore)
      .filter((score): score is number => Boolean(score && score > 0));
    if (!scores.length) {
      return 5.5;
    }
    return scores.reduce((sum, value) => sum + value, 0) / scores.length;
  }, [activeEntries]);

  useEffect(() => {
    if (!done || scoreMeanTarget !== null) {
      return;
    }
    setScoreMeanTarget(Number(malScoreMean.toFixed(2)));
  }, [done, scoreMeanTarget, malScoreMean]);

  const priorityData = useMemo(() => {
    const scoreById = new Map<number, number | null>();
    const scores: number[] = [];
    let completedCount = 0;
    let droppedCount = 0;

    activeEntries.forEach((entry) => {
      scoreById.set(entry.animeId, entry.myScore ?? null);
      if (entry.myScore && entry.myScore > 0) {
        scores.push(entry.myScore);
      }
      if (entry.status === "Completed") {
        completedCount += 1;
      } else if (entry.status === "Dropped") {
        droppedCount += 1;
      }
    });

    const priorityById = new Map<number, number>();
    if (!scores.length) {
      activeEntries.forEach((entry) => priorityById.set(entry.animeId, 0));
      return {
        scoreById,
        priorityById,
        priorityBoost: 0,
        sameScoreBoost: 0,
      };
    }

    const fit = fitNormal(scores);
    const total = scores.length;
    const observed = Array.from({ length: 10 }, (_, idx) => {
      const score = idx + 1;
      return scores.filter((s) => s === score).length / total;
    });

    let expected = Array.from({ length: 10 }, () => 0);
    if (fit.sigma > 0) {
      expected = Array.from({ length: 10 }, (_, idx) => {
        const score = idx + 1;
        const low = score - 0.5;
        const high = score + 0.5;
        return (
          normalCdf((high - fit.mu) / fit.sigma) -
          normalCdf((low - fit.mu) / fit.sigma)
        );
      });
    }

    const deviations = observed.map((value, idx) =>
      Math.max(0, value - expected[idx]),
    );
    const maxDeviation = Math.max(...deviations, 0);
    const priorityByScore = deviations.map((value) =>
      maxDeviation > 0 ? value / maxDeviation : 0,
    );

    activeEntries.forEach((entry) => {
      const score = entry.myScore ?? 0;
      const priority = score ? priorityByScore[score - 1] : 0;
      priorityById.set(entry.animeId, priority);
    });

    let normality = 0.5;
    if (scores.length < 10) {
      normality = fit.sigma <= 0 ? 0 : 0.5;
    } else if (fit.sigma <= 0) {
      normality = 0;
    } else {
      const l1 = observed.reduce(
        (sum, value, idx) => sum + Math.abs(value - expected[idx]),
        0,
      );
      normality = Math.max(0, Math.min(1, 1 - l1 / 2));
    }

    const n = activeEntries.length;
    const totalPairs = (n * (n - 1)) / 2;
    let pairScale = 1;
    if (
      totalPairs > 0 &&
      appliedAssumeDroppedLower &&
      completedCount &&
      droppedCount
    ) {
      const allowedPairs = totalPairs - completedCount * droppedCount;
      pairScale = allowedPairs / totalPairs;
    }
    const normalityMultiplier = Math.max(
      0.35,
      Math.min(1.2, 1.2 - 0.9 * normality),
    );
    const comparisonScale = pairScale * normalityMultiplier;
    const priorityGuidance = comparisonTargets(n, comparisonScale);
    const optimal = Math.max(1, priorityGuidance.optimal);
    const target = Math.max(1, comparisonGoal);
    const scarcity = Math.min(2.5, optimal / target);
    const normalityGap = 1 - normality;
    const priorityBoost = Math.min(
      2.5,
      Math.max(0, (scarcity - 1) * 1.2 + normalityGap),
    );

    return {
      scoreById,
      priorityById,
      priorityBoost,
      sameScoreBoost: priorityBoost * 0.75,
    };
  }, [activeEntries, appliedAssumeDroppedLower, comparisonGoal]);

  const pairAllowed = useCallback(
    (aId: number, bId: number, assumeLower: boolean) => {
      if (!assumeLower) {
        return true;
      }
      const aStatus = statusById.get(aId);
      const bStatus = statusById.get(bId);
      return !(
        (aStatus === "Completed" && bStatus === "Dropped") ||
        (aStatus === "Dropped" && bStatus === "Completed")
      );
    },
    [statusById],
  );

  const selectAllowedPair = useCallback(
    (
      elo: EloState,
      animeIds: number[],
      avoid: boolean,
      assumeLower: boolean,
    ): [number, number] | null => {
      for (let attempt = 0; attempt < 250; attempt += 1) {
        const [aId, bId] = selectPair(elo, animeIds, {
          rng: rngRef.current,
          avoidRepeats: avoid,
          priorityById: priorityData.priorityById,
          priorityBoost: priorityData.priorityBoost,
          scoreById: priorityData.scoreById,
          sameScoreBoost: priorityData.sameScoreBoost,
        });
        if (pairAllowed(aId, bId, assumeLower)) {
          return [aId, bId];
        }
      }

      const allowed: [number, number][] = [];
      for (let i = 0; i < animeIds.length; i += 1) {
        for (let j = i + 1; j < animeIds.length; j += 1) {
          if (pairAllowed(animeIds[i], animeIds[j], assumeLower)) {
            allowed.push([animeIds[i], animeIds[j]]);
          }
        }
      }
      if (!allowed.length) {
        return null;
      }
      const idx = Math.floor(rngRef.current.nextFloat() * allowed.length);
      return allowed[idx];
    },
    [pairAllowed, priorityData],
  );

  useEffect(() => {
    if (
      !settingsConfirmed ||
      !eloState ||
      currentPair ||
      !appliedAnimeIds.length
    ) {
      return;
    }
    const pair = selectAllowedPair(
      eloState,
      appliedAnimeIds,
      appliedSettings?.avoidRepeats ?? avoidRepeats,
      appliedSettings?.assumeDroppedLower ?? assumeDroppedLower,
    );
    if (!pair) {
      setPairError("No valid pairs available with current assumptions.");
      return;
    }
    setCurrentPair(pair);
  }, [
    settingsConfirmed,
    eloState,
    currentPair,
    appliedAnimeIds,
    avoidRepeats,
    assumeDroppedLower,
    appliedSettings,
    selectAllowedPair,
  ]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    if (!exportData) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }
    const payload: StoredState = {
      version: 1,
      exportData,
      fileName,
      statusFilter,
      showPosters,
      showEnglish,
      assumeDroppedLower,
      avoidRepeats,
      kFactor,
      comparisonsTarget,
      settingsConfirmed,
      appliedSettings,
      appliedAnimeIds,
      scoreMeanTarget,
      autoDecisions,
      currentPair,
      eloState: eloState ? serializeEloState(eloState) : null,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }, [
    hydrated,
    exportData,
    fileName,
    statusFilter,
    showPosters,
    showEnglish,
    assumeDroppedLower,
    avoidRepeats,
    kFactor,
    comparisonsTarget,
    settingsConfirmed,
    appliedSettings,
    appliedAnimeIds,
    scoreMeanTarget,
    autoDecisions,
    currentPair,
    eloState,
  ]);

  const currentPairEntries = useMemo(() => {
    if (!currentPair || !exportData) {
      return null;
    }
    return currentPair.map((id) => animeById.get(id) as AnimeEntry);
  }, [currentPair, exportData, animeById]);

  const skipDecision = useMemo<SkipDecision | null>(() => {
    if (!currentPairEntries) {
      return null;
    }
    const [left, right] = currentPairEntries;
    const leftStatus = left.status ?? "";
    const rightStatus = right.status ?? "";
    const leftScore = left.myScore ?? 0;
    const rightScore = right.myScore ?? 0;
    if (leftStatus === rightStatus && leftScore === rightScore) {
      return null;
    }
    if (leftScore !== rightScore) {
      return {
        outcome: leftScore > rightScore ? 1 : 0,
        reason: "Higher MAL score wins",
      };
    }
    return { outcome: 0.5, reason: "Same MAL score" };
  }, [currentPairEntries]);

  const applyOutcome = useCallback(
    (outcome: number) => {
      if (!eloState || !currentPair) {
        return;
      }
      const updated = recordOutcome(
        eloState,
        currentPair[0],
        currentPair[1],
        outcome,
      );
      const pair = selectAllowedPair(
        updated,
        appliedAnimeIds,
        appliedAvoidRepeats,
        appliedAssumeDroppedLower,
      );
      setEloState(updated);
      if (!pair) {
        setCurrentPair(null);
        setPairError("No valid pairs available with current assumptions.");
        return;
      }
      setCurrentPair(pair);
    },
    [
      eloState,
      currentPair,
      appliedAnimeIds,
      appliedAvoidRepeats,
      appliedAssumeDroppedLower,
      selectAllowedPair,
    ],
  );

  const applyAutoSkip = useCallback(() => {
    if (!skipDecision) {
      return;
    }
    setAutoDecisions((prev) => prev + 1);
    applyOutcome(skipDecision.outcome);
  }, [skipDecision, applyOutcome]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!currentPair || done) {
        return;
      }
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "a") {
        event.preventDefault();
        applyOutcome(1);
      } else if (key === "d") {
        event.preventDefault();
        applyOutcome(0);
      } else if (key === "s") {
        if (skipDecision) {
          event.preventDefault();
          applyAutoSkip();
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [currentPair, done, applyOutcome, skipDecision, applyAutoSkip]);

  const topRows = useMemo(() => {
    if (!eloState) {
      return [];
    }
    const statusTier: Record<string, number> | undefined =
      appliedAssumeDroppedLower ? { Dropped: -1 } : undefined;
    const rows = Array.from(eloState.ratings.entries()).sort((left, right) => {
      const [leftId, leftRating] = left;
      const [rightId, rightRating] = right;
      const leftStatus = animeById.get(leftId)?.status || "";
      const rightStatus = animeById.get(rightId)?.status || "";
      const leftTier = statusTier ? (statusTier[leftStatus] ?? 0) : 0;
      const rightTier = statusTier ? (statusTier[rightStatus] ?? 0) : 0;
      if (leftTier !== rightTier) {
        return rightTier - leftTier;
      }
      return rightRating.value - leftRating.value;
    });
    return rows.slice(0, 15).map(([animeId, rating]) => {
      const entry = animeById.get(animeId) as AnimeEntry;
      return {
        title: entry.title,
        elo: rating.value,
        games: rating.games,
        wins: rating.wins,
        losses: rating.losses,
        ties: rating.ties,
      };
    });
  }, [eloState, animeById, appliedAssumeDroppedLower]);

  const finalFit = useMemo(() => {
    if (!eloState) {
      return null;
    }
    return fitNormal(Array.from(eloState.ratings.values()).map((r) => r.value));
  }, [eloState]);

  const basePercentiles = useMemo(() => {
    if (!eloState || !finalFit) {
      return [];
    }
    return Array.from(eloState.ratings.values()).map((rating) =>
      percentile(rating.value, finalFit),
    );
  }, [eloState, finalFit]);

  const scoreMeanValue = scoreMeanTarget ?? Number(malScoreMean.toFixed(2));

  const percentileShift = useMemo(() => {
    if (!basePercentiles.length) {
      return 0;
    }
    const target = Math.max(1, Math.min(10, scoreMeanValue));
    const meanForShift = (shift: number) => {
      const total = basePercentiles.reduce((sum, value) => {
        const adjusted = Math.min(1, Math.max(0, value + shift));
        return sum + score10FromPercentile(adjusted);
      }, 0);
      return total / basePercentiles.length;
    };
    let low = -1;
    let high = 1;
    let bestShift = 0;
    let bestDiff = Number.POSITIVE_INFINITY;
    for (let i = 0; i < 30; i += 1) {
      const mid = (low + high) / 2;
      const mean = meanForShift(mid);
      const diff = mean - target;
      const absDiff = Math.abs(diff);
      if (absDiff < bestDiff) {
        bestDiff = absDiff;
        bestShift = mid;
      }
      if (diff < 0) {
        low = mid;
      } else {
        high = mid;
      }
    }
    return bestShift;
  }, [basePercentiles, scoreMeanValue]);

  const appliedExcessive = useMemo(() => {
    const n = activeAnimeIds.length;
    if (n <= 0) return 1;
    const totalPairs = (n * (n - 1)) / 2;
    let pairScale = 1;
    if (totalPairs > 0 && appliedAssumeDroppedLower) {
      const completed = activeEntries.filter(
        (e) => e.status === "Completed",
      ).length;
      const dropped = activeEntries.filter(
        (e) => e.status === "Dropped",
      ).length;
      if (completed && dropped) {
        const allowedPairs = totalPairs - completed * dropped;
        pairScale = allowedPairs / totalPairs;
      }
    }
    const normMult = Math.max(
      0.35,
      Math.min(1.2, 1.2 - 0.9 * appliedMalNormality),
    );
    const scale = pairScale * normMult;
    return comparisonTargets(n, scale).excessive;
  }, [
    activeAnimeIds.length,
    activeEntries,
    appliedAssumeDroppedLower,
    appliedMalNormality,
  ]);

  const excessiveRatio = useMemo(() => {
    if (!eloState || appliedExcessive <= 0) return 0;
    return Math.min(1, eloState.comparisons / appliedExcessive);
  }, [eloState, appliedExcessive]);

  const blendingParams = useMemo<BlendingParams | undefined>(() => {
    if (!malSummary || !eloState) {
      return undefined;
    }
    return {
      malNormality: appliedMalNormality,
      completionRatio: excessiveRatio,
      malFit: malSummary.fit,
      totalComparisons: eloState.comparisons,
      itemCount: activeAnimeIds.length,
    };
  }, [
    appliedMalNormality,
    excessiveRatio,
    malSummary,
    eloState,
    activeAnimeIds.length,
  ]);

  const results = useMemo<AnimeResult[] | null>(() => {
    if (!eloState || !finalFit) {
      return null;
    }
    const statusTier = appliedAssumeDroppedLower ? { Dropped: -1 } : undefined;
    return buildResults(
      animeById,
      eloState,
      finalFit,
      statusTier,
      percentileShift,
      blendingParams,
    );
  }, [
    eloState,
    finalFit,
    animeById,
    appliedAssumeDroppedLower,
    percentileShift,
    blendingParams,
  ]);

  const eloWeight = computeEloWeight(excessiveRatio, appliedMalNormality);

  const resultsMetadata = useMemo(() => {
    if (!exportData || !eloState || !finalFit) {
      return null;
    }
    const statusTier = appliedAssumeDroppedLower ? { Dropped: -1 } : null;
    return {
      user_name: exportData.userName,
      user_id: exportData.userId,
      source_filename: fileName,
      included_statuses: appliedStatuses,
      comparisons: eloState.comparisons,
      skipped: eloState.skips,
      elo_k_factor: appliedKFactor,
      elo_initial_rating: 1500,
      normal_fit_mu: finalFit.mu,
      normal_fit_sigma: finalFit.sigma,
      score_mean_target: scoreMeanValue,
      percentile_shift: percentileShift,
      assume_dropped_lower: appliedAssumeDroppedLower,
      status_tier: statusTier,
      mal_score_normality: appliedMalNormality,
      normality_multiplier: appliedNormalityMultiplier,
      elo_weight: eloWeight,
      mal_weight: 1 - eloWeight,
    };
  }, [
    exportData,
    eloState,
    finalFit,
    fileName,
    appliedStatuses,
    appliedKFactor,
    scoreMeanValue,
    percentileShift,
    appliedAssumeDroppedLower,
    appliedMalNormality,
    appliedNormalityMultiplier,
    eloWeight,
  ]);

  const handleSaveToDisk = useCallback(async () => {
    if (!results || !resultsMetadata) {
      return;
    }
    if (!window.showDirectoryPicker) {
      alert(
        "Directory access is not supported in this browser. Use downloads instead.",
      );
      return;
    }
    const base = safeBaseName(`anime_ranker_${exportData?.userName || "user"}`);
    const dir = await window.showDirectoryPicker();
    const csvHandle = await dir.getFileHandle(`${base}.csv`, { create: true });
    const jsonHandle = await dir.getFileHandle(`${base}.json`, {
      create: true,
    });

    const csvWritable = await csvHandle.createWritable();
    await csvWritable.write(resultsToCsv(results));
    await csvWritable.close();

    const jsonWritable = await jsonHandle.createWritable();
    await jsonWritable.write(resultsToJson(resultsMetadata, results));
    await jsonWritable.close();
  }, [results, resultsMetadata, exportData]);

  const leftEntry = currentPairEntries?.[0];
  const rightEntry = currentPairEntries?.[1];
  const leftMedia = leftEntry ? mediaCache.get(leftEntry.animeId) : null;
  const rightMedia = rightEntry ? mediaCache.get(rightEntry.animeId) : null;

  const hero = (
    <header className="hero">
      <div>
        <p className="eyebrow">Elo Anime Ranker</p>
        <h1>Rank your anime with pairwise comparisons</h1>
        <p className="lede">
          Upload a MAL export, smash hotkeys, and let Elo build a balanced
          ranking in minutes.
        </p>
      </div>
    </header>
  );

  return (
    <div className="app">
      {!inCombat && hero}

      <div className="layout">
        <aside className="sidebar">
          <div className="panel">
            <h2>Session setup</h2>
            <label className="upload">
              <input
                type="file"
                accept=".xml,.gz"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void handleFile(file);
                  }
                }}
              />
              <span>Upload MAL export</span>
              <em>{fileName ?? "XML or XML.GZ"}</em>
            </label>
            {exportData && (
              <div className="user-meta">
                <span>
                  <strong>{exportData.userName || "unknown"}</strong>
                </span>
                <span>{exportData.anime.length} anime</span>
                <span>
                  <strong>{selectedEntries.length}</strong> eligible
                </span>
              </div>
            )}
            {!exportData && (
              <p className="muted">Upload your MyAnimeList export to begin.</p>
            )}
            {exportData && (
              <>
                {selectedEntries.length < 2 && (
                  <p className="warning">
                    Select at least 2 anime (adjust status filters).
                  </p>
                )}

                <div className="section">
                  <p className="section-title">Comparisons target</p>
                  <div className="range-meta">
                    <span className="range-current" style={currentValueStyle}>
                      {comparisonsTarget}
                    </span>
                    <span className="range-max">{sliderMax}</span>
                  </div>
                  <input
                    type="range"
                    className="range-input"
                    min={0}
                    max={sliderMax}
                    value={Math.min(comparisonsTarget, sliderMax)}
                    onChange={(event) =>
                      setComparisonsTarget(Number(event.target.value))
                    }
                  />
                  <div className="range-marks">
                    {rangeMarks.map((mark) => (
                      <div
                        key={mark.label}
                        className="range-mark"
                        style={rangeMarkerStyle(mark.value)}
                      >
                        <span className="range-mark-line" />
                        <span className="range-mark-label">{mark.label}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="section">
                  <p className="section-title">MAL score shape</p>
                  <div className="slider-readonly">
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={Math.round(malSummary.normality * 100)}
                      disabled
                    />
                    <span>{Math.round(malSummary.normality * 100)}%</span>
                  </div>
                  <NormalityChart
                    scores={malSummary.scores}
                    mu={malSummary.fit.mu}
                    sigma={malSummary.fit.sigma}
                  />
                </div>

                <button
                  className="primary start-button"
                  onClick={startRanking}
                  disabled={selectedEntries.length < 2}
                >
                  {settingsConfirmed ? "Reset ranking" : "Start ranking"}
                </button>

                <details className="section">
                  <summary>Advanced</summary>
                  <div className="advanced">
                    <p className="section-title">Filters</p>
                    <div className="checkbox-grid">
                      {allStatuses.map((status) => (
                        <label key={status} className="checkbox">
                          <input
                            type="checkbox"
                            checked={statusFilter.includes(status)}
                            onChange={(event) => {
                              setStatusFilter((prev) => {
                                const set = new Set(prev);
                                if (event.target.checked) {
                                  set.add(status);
                                } else {
                                  set.delete(status);
                                }
                                return Array.from(set);
                              });
                            }}
                          />
                          <span>{status}</span>
                        </label>
                      ))}
                    </div>
                    <p className="section-title">Visual aids</p>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={showPosters}
                        onChange={(event) =>
                          setShowPosters(event.target.checked)
                        }
                      />
                      <span>Show posters (Jikan)</span>
                    </label>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={showEnglish}
                        onChange={(event) =>
                          setShowEnglish(event.target.checked)
                        }
                      />
                      <span>Show English titles</span>
                    </label>
                    <p className="section-title">Assumptions</p>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={assumeDroppedLower}
                        onChange={(event) =>
                          setAssumeDroppedLower(event.target.checked)
                        }
                      />
                      <span>Dropped is worse than Completed</span>
                    </label>
                    <p className="section-title">Behavior</p>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={avoidRepeats}
                        onChange={(event) =>
                          setAvoidRepeats(event.target.checked)
                        }
                      />
                      <span>Avoid repeating the same pair</span>
                    </label>
                    <p className="section-title">Elo</p>
                    <label className="slider">
                      <span>K-factor: {kFactor.toFixed(0)}</span>
                      <input
                        type="range"
                        min={1}
                        max={200}
                        step={1}
                        value={kFactor}
                        onChange={(event) =>
                          setKFactor(Number(event.target.value))
                        }
                      />
                    </label>
                  </div>
                </details>
              </>
            )}
          </div>
        </aside>

        <main className="main">
          {!exportData && (
            <div className="panel empty">
              <p>Upload a MAL export to unlock the ranking flow.</p>
            </div>
          )}
          {exportData && !settingsConfirmed && (
            <div className="panel">
              <h2>Ready when you are</h2>
              <p>
                Choose settings in the sidebar, then click Start ranking to
                apply.
              </p>
            </div>
          )}
          {exportData && settingsConfirmed && !eloState && (
            <div className="panel">
              <p>Ranking not initialized. Click Start ranking.</p>
            </div>
          )}
          {exportData && settingsConfirmed && eloState && (
            <div className="stack">
              {pairError && <div className="panel error">{pairError}</div>}

              {!done && currentPairEntries && (
                <div className="panel compare">
                  <h2>What&apos;s cooler?</h2>
                  <div className="compare-grid">
                    {[leftEntry, rightEntry].map((entry, idx) => {
                      if (!entry) {
                        return null;
                      }
                      const media = idx === 0 ? leftMedia : rightMedia;
                      const key = idx === 0 ? "left" : "right";
                      return (
                        <div key={key} className="compare-card">
                          {showPosters && media?.imageUrl && (
                            <img src={media.imageUrl} alt="" />
                          )}
                          <button
                            className="vote"
                            onClick={() => applyOutcome(idx === 0 ? 1 : 0)}
                          >
                            [{idx === 0 ? "A" : "D"}] {entry.title}
                          </button>
                          <div className="meta">
                            {showEnglish && media?.titleEnglish && (
                              <span>EN: {media.titleEnglish}</span>
                            )}
                            {entry.status && <span>{entry.status}</span>}
                            <span>MAL {entry.myScore ?? "-"}/10</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="actions">
                    <button
                      className="secondary"
                      onClick={applyAutoSkip}
                      disabled={!skipDecision}
                      title={
                        skipDecision
                          ? skipDecision.reason
                          : "Auto-decide from MAL (disabled only when status and score match)."
                      }
                    >
                      [S] Skip (use MAL)
                    </button>
                    <button
                      className="secondary"
                      onClick={() => setComparisonsTarget(eloState.comparisons)}
                    >
                      Finish now
                    </button>
                  </div>
                </div>
              )}

              <div className="panel">
                <div className="progress-row">
                  <progress value={progressValue} max={1} />
                  <span>
                    Comparisons: {eloState.comparisons} / {comparisonGoal}
                  </span>
                </div>
              </div>

              {!done && topRows.length > 0 && (
                <div className="panel">
                  <h2>Current top 15</h2>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Title</th>
                        <th>Elo</th>
                        <th>Games</th>
                        <th>W</th>
                        <th>L</th>
                        <th>T</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topRows.map((row) => (
                        <tr key={row.title}>
                          <td>{row.title}</td>
                          <td>{row.elo.toFixed(1)}</td>
                          <td>{row.games}</td>
                          <td>{row.wins}</td>
                          <td>{row.losses}</td>
                          <td>{row.ties}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="hint">
                    Reach the comparison target to compute the 1-10 scores and
                    export.
                  </p>
                </div>
              )}

              {done && results && resultsMetadata && finalFit && (
                <div className="panel">
                  <h2>Elo to fitted Normal to 1-10</h2>
                  <div
                    className="scoring-math"
                    ref={(el) => {
                      if (el && window.MathJax?.typesetPromise) {
                        window.MathJax.typesetPromise([el]);
                      }
                    }}
                  >
                    <p className="math-line">
                      {"$\\mathcal{N}(\\mu=" +
                        finalFit.mu.toFixed(1) +
                        ",\\, \\sigma=" +
                        finalFit.sigma.toFixed(1) +
                        ") \\cdot " +
                        Math.round(eloWeight * 100) +
                        "\\%\\text{ Elo} + " +
                        Math.round((1 - eloWeight) * 100) +
                        "\\%\\text{ MAL}" +
                        (excessiveRatio < 1
                          ? " \\cdot " +
                            (eloState?.comparisons ?? 0) +
                            "/" +
                            appliedExcessive +
                            "\\text{ comparisons}"
                          : "") +
                        "$"}
                    </p>
                    <p className="hint math-explain">
                      Fewer comparisons or better MAL distribution → trust MAL
                      more. Items with more games have more reliable Elo
                      ratings.
                    </p>
                    <div className="math-grid">
                      <div className="math-item">
                        <span className="math-label">Global weight</span>
                        <span>{"$w = r^{1+(1-n)}$"}</span>
                      </div>
                      <div className="math-item">
                        <span className="math-label">Item confidence</span>
                        <span>{"$c_i = 1 - e^{-g_i/\\bar{g}}$"}</span>
                      </div>
                      <div className="math-item">
                        <span className="math-label">Blend</span>
                        <span>
                          {
                            "$p = w c_i \\cdot p_{\\text{elo}} + (1 - w c_i) \\cdot p_{\\text{mal}}$"
                          }
                        </span>
                      </div>
                      <div className="math-item">
                        <span className="math-label">Score</span>
                        <span>{"$\\lceil 10p \\rceil$"}</span>
                      </div>
                    </div>
                    <div className="bell-curve-container">
                      {(() => {
                        const sorted = [...results].sort(
                          (a, b) => a.percentile - b.percentile,
                        );
                        const worst = sorted[0];
                        const best = sorted[sorted.length - 1];
                        const midIdx = Math.floor(sorted.length / 2);
                        const middle = sorted[midIdx];
                        const samples = [
                          { ...worst, label: "worst", color: "#c44" },
                          { ...middle, label: "mid", color: "#888" },
                          { ...best, label: "best", color: "#4a4" },
                        ];
                        const width = 260;
                        const height = 80;
                        const pad = { l: 25, r: 10, t: 25, b: 18 };
                        const chartW = width - pad.l - pad.r;
                        const chartH = height - pad.t - pad.b;
                        // X axis is score 1-10
                        const scaleX = (score: number) =>
                          pad.l + ((score - 1) / 9) * chartW;
                        // Bell curve centered at target mean
                        const targetMean = scoreMeanValue;
                        const spread = 2.5; // standard deviation in score units
                        const gaussian = (score: number) => {
                          const z = (score - targetMean) / spread;
                          return Math.exp(-0.5 * z * z);
                        };
                        const scaleY = (g: number) => pad.t + (1 - g) * chartH;
                        const steps = 50;
                        const pathPoints = Array.from(
                          { length: steps + 1 },
                          (_, i) => {
                            const score = 1 + (i / steps) * 9;
                            return `${i === 0 ? "M" : "L"}${scaleX(score).toFixed(1)},${scaleY(gaussian(score)).toFixed(1)}`;
                          },
                        ).join(" ");
                        const scoreTicks = [1, 5, 10];
                        return (
                          <svg
                            className="elo-bell-curve"
                            viewBox={`0 0 ${width} ${height}`}
                          >
                            <path className="bell-path" d={pathPoints} />
                            <line
                              className="bell-axis"
                              x1={pad.l}
                              y1={scaleY(0)}
                              x2={width - pad.r}
                              y2={scaleY(0)}
                            />
                            {scoreTicks.map((score) => {
                              const x = pad.l + ((score - 1) / 9) * chartW;
                              return (
                                <text
                                  key={score}
                                  className="bell-tick"
                                  x={x}
                                  y={height - 2}
                                >
                                  {score}
                                </text>
                              );
                            })}
                            {samples.map((s) => {
                              const sampleScore = s.score1to10;
                              const x = Math.max(
                                pad.l + 5,
                                Math.min(
                                  width - pad.r - 5,
                                  scaleX(sampleScore),
                                ),
                              );
                              const y = scaleY(gaussian(sampleScore));
                              const shortTitle =
                                s.title.length > 12
                                  ? s.title.slice(0, 11) + "…"
                                  : s.title;
                              return (
                                <g key={s.animeId}>
                                  <circle cx={x} cy={y} r={3} fill={s.color} />
                                  <text
                                    x={x}
                                    y={y - 6}
                                    className="bell-sample"
                                    fill={s.color}
                                  >
                                    {shortTitle}
                                  </text>
                                </g>
                              );
                            })}
                          </svg>
                        );
                      })()}
                    </div>
                  </div>
                  <div className="mean-control">
                    <div className="mean-header">
                      <span>Target mean for final scores</span>
                      <strong>{scoreMeanValue.toFixed(2)}</strong>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={10}
                      step={0.01}
                      value={scoreMeanValue}
                      onChange={(event) =>
                        setScoreMeanTarget(Number(event.target.value))
                      }
                    />
                  </div>
                  <div className="results-actions">
                    <button
                      className="primary"
                      onClick={() =>
                        downloadBlob(
                          "anime_ranker_results.csv",
                          resultsToCsv(results),
                          "text/csv",
                        )
                      }
                    >
                      Download CSV
                    </button>
                    <button
                      className="primary"
                      onClick={() =>
                        downloadBlob(
                          "anime_ranker_results.json",
                          resultsToJson(resultsMetadata, results),
                          "application/json",
                        )
                      }
                    >
                      Download JSON
                    </button>
                    <button className="secondary" onClick={handleSaveToDisk}>
                      Save to disk
                    </button>
                  </div>
                  <table className="data-table results">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Title</th>
                        <th>Status</th>
                        <th>MAL</th>
                        <th>Elo</th>
                        <th>Games</th>
                        <th>Percentile</th>
                        <th>Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((row) => (
                        <tr key={row.animeId}>
                          <td>{row.rank}</td>
                          <td>{row.title}</td>
                          <td>{row.status ?? "-"}</td>
                          <td>{row.myScore ?? "-"}</td>
                          <td>{row.elo.toFixed(1)}</td>
                          <td>{row.games}</td>
                          <td>{row.percentile.toFixed(4)}</td>
                          <td>{row.score1to10}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
