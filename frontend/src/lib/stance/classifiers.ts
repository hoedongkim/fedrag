/**
 * The four stance classifiers: display names, descriptions and chart colors. Each classifier owns
 * a fixed categorical color slot (--series-1..4 in globals.css), so toggling one never repaints
 * the others.
 */
import type { Stance, StanceClassifier } from "../api/types";

export interface ClassifierInfo {
  id: StanceClassifier;
  label: string;
  /** One line on how the classifier works. */
  description: string;
  /** CSS color of its line and key, e.g. "var(--series-1)". */
  color: string;
  /** True when score and stance come from sentence labels (score = mean P(hawkish) - P(dovish)). */
  sentenceBased: boolean;
}

/** In display order; the slot order is the validated palette order (see globals.css). */
export const CLASSIFIERS: readonly ClassifierInfo[] = [
  {
    id: "llm_few_shot",
    label: "LLM few-shot",
    description: "GPT-5.4 mini reads the whole statement, with labelled example sentences.",
    color: "var(--series-1)",
    sentenceBased: false,
  },
  {
    id: "llm_zero_shot",
    label: "LLM zero-shot",
    description: "GPT-5.4 mini reads the whole statement, with the rubric only.",
    color: "var(--series-2)",
    sentenceBased: false,
  },
  {
    id: "finbert",
    label: "FinBERT",
    description: "FinBERT fine-tuned on Trillion Dollar Words, averaged over its sentences.",
    color: "var(--series-3)",
    sentenceBased: true,
  },
  {
    id: "keyword",
    label: "Keyword",
    description: "A transparent hawkish/dovish word list with negation handling.",
    color: "var(--series-4)",
    sentenceBased: true,
  },
];

export const CLASSIFIER_IDS: readonly StanceClassifier[] = CLASSIFIERS.map((c) => c.id);
export const DEFAULT_CLASSIFIER: StanceClassifier = "llm_few_shot";

export function classifierInfo(id: StanceClassifier): ClassifierInfo {
  return CLASSIFIERS.find((c) => c.id === id) ?? CLASSIFIERS[0];
}

export function isClassifier(value: string | null | undefined): value is StanceClassifier {
  return CLASSIFIER_IDS.some((id) => id === value);
}

/** Stances from dovish to hawkish, the order of the score axis. */
export const STANCES: readonly Stance[] = ["dovish", "neutral", "hawkish"];

/** Sentence classifiers turn their mean score into a stance with these fixed cut-offs. */
export const SENTENCE_THRESHOLD = 0.15;

/**
 * Stroke dash of the sentence classifiers' lines and keys. A second encoding besides color: with
 * all four lines on, the LLM zero-shot and Keyword colors are too close to tell apart reliably.
 */
export function seriesDash(info: Pick<ClassifierInfo, "sentenceBased">): string | undefined {
  return info.sentenceBased ? "6 3" : undefined;
}
