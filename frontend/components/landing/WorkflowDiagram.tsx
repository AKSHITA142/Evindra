"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  BrainCircuit,
  ClipboardList,
  PlayCircle,
  FlaskConical,
  CheckCircle2,
  GitFork,
  ArrowRight,
  ArrowDown,
  RotateCcw,
  Sparkles,
  ShieldCheck,
  Zap,
  Play,
  Pause,
  ChevronRight,
  Code2,
  Layers,
  Database,
  Activity,
} from "lucide-react";

interface StageInfo {
  id: string;
  step: string;
  label: string;
  nodeName: string;
  tagline: string;
  icon: React.ElementType;
  color: string;
  coreOutputs: string[];
  metricsPreview: { name: string; val: string }[];
  statePayloadSnippet: Record<string, unknown>;
  visualComponentType: "profile" | "hypotheses" | "cv" | "radar" | "lift" | "loop";
}

const STAGES: StageInfo[] = [
  {
    id: "understand",
    step: "01",
    label: "Semantic Understanding",
    nodeName: "node_understand()",
    tagline: "Profile schema, data quality, target distribution, and drop leakage IDs.",
    icon: BrainCircuit,
    color: "text-brand-300",
    coreOutputs: [
      "Target classification vs regression detected",
      "Leakage IDs & timestamp indices auto-dropped",
      "SemanticProfile & quality constraints saved to state",
    ],
    metricsPreview: [
      { name: "Leakage IDs Removed", val: "100% (2 cols)" },
      { name: "Target Imbalance", val: "68% / 32%" },
    ],
    statePayloadSnippet: {
      target_column: "churn_risk",
      task_type: "binary_classification",
      dataset_shape: [12500, 18],
      dropped_leakage_cols: ["user_id", "created_at"],
      imbalance_ratio: 0.32,
      profile_valid: true,
    },
    visualComponentType: "profile",
  },
  {
    id: "plan",
    step: "02",
    label: "Hypothesis Formulation",
    nodeName: "node_plan()",
    tagline: "Generate prioritized strategies, feature transformations, and model bounds.",
    icon: ClipboardList,
    color: "text-info-400",
    coreOutputs: [
      "Candidate hypotheses ranked by expected information gain",
      "Encoders & scalers paired per feature type",
      "Model family hyperparameter search space configured",
    ],
    metricsPreview: [
      { name: "Hypotheses Generated", val: "4 Candidates" },
      { name: "Top Strategy", val: "CatBoost + TargetEnc" },
    ],
    statePayloadSnippet: {
      hypotheses_queue: [
        { id: "H-01", encoder: "TargetEncoder", model: "CatBoost", expected_gain: 0.94 },
        { id: "H-02", encoder: "QuantileTransformer", model: "XGBoost", expected_gain: 0.91 },
        { id: "H-03", encoder: "OneHotEncoder", model: "LightGBM", expected_gain: 0.88 },
      ],
      current_iteration: 1,
      max_budget: 5,
    },
    visualComponentType: "hypotheses",
  },
  {
    id: "execute",
    step: "03",
    label: "Leakage-Safe Execution",
    nodeName: "node_execute()",
    tagline: "Build reproducible pipelines & train models with strict 5-fold cross validation.",
    icon: PlayCircle,
    color: "text-warning-400",
    coreOutputs: [
      "Fit-transform isolated within training folds only",
      "XGBoost, LightGBM, CatBoost & Random Forest trained",
      "Out-of-fold validation predictions recorded",
    ],
    metricsPreview: [
      { name: "Validation Strategy", val: "5-Fold Stratified" },
      { name: "CV Train Latency", val: "3.42s" },
    ],
    statePayloadSnippet: {
      active_pipeline: "CatBoost_TargetEnc_Pipeline",
      cv_folds_evaluated: "5/5",
      mean_validation_auc: 0.938,
      fold_stdev: 0.012,
      inference_latency_ms: 1.45,
      leakage_check_passed: true,
    },
    visualComponentType: "cv",
  },
  {
    id: "evaluate",
    step: "04",
    label: "4-Pillar Evaluation",
    nodeName: "node_evaluate()",
    tagline: "Score candidates across Generalization, Stability, Efficiency, and Business Fit.",
    icon: FlaskConical,
    color: "text-success-400",
    coreOutputs: [
      "Generalization (35%) + Fold Stability (25%) computed",
      "Inference Latency (20%) + Resource Fit (20%) factored",
      "Overfitting gap calculated across train vs test splits",
    ],
    metricsPreview: [
      { name: "Mean Fold AUC", val: "0.938 ± 0.012" },
      { name: "Composite Rank", val: "Score: 0.912" },
    ],
    statePayloadSnippet: {
      composite_score: 0.924,
      score_weights: {
        generalization_35: 0.938,
        stability_25: 0.965,
        efficiency_20: 0.890,
        business_fit_20: 0.910,
      },
      overfitting_gap: 0.018,
      rank: 1,
    },
    visualComponentType: "radar",
  },
  {
    id: "direct",
    step: "05",
    label: "Evidence Synthesis",
    nodeName: "node_direct()",
    tagline: "Promote winning hypotheses, log discoveries, and update research memory.",
    icon: CheckCircle2,
    color: "text-brand-300",
    coreOutputs: [
      "Research knowledge base updated with feature impacts",
      "Champion model weights & pipeline artifact frozen",
      "Executive explanation generated for stakeholders",
    ],
    metricsPreview: [
      { name: "Hypothesis Lift", val: "+1.84% Normalized" },
      { name: "Knowledge Memory", val: "3 Winning Insights" },
    ],
    statePayloadSnippet: {
      champion_model: "CatBoostClassifier_v1",
      promoted_hypotheses: ["H-01: TargetEnc on categorical features yielded +1.84% lift"],
      knowledge_base_updated: true,
      artifact_ready: "champion_pipeline.joblib",
    },
    visualComponentType: "lift",
  },
  {
    id: "route",
    step: "06",
    label: "Adaptive Routing / Ship",
    nodeName: "route_decision()",
    tagline: "If gain > 0.5% & budget remains, loop to Planner; otherwise package & ship.",
    icon: GitFork,
    color: "text-info-400",
    coreOutputs: [
      "Evaluates research budget (Iteration 1 of 5)",
      "Threshold check: delta > 0.5% triggers next research cycle",
      "Final state compiles downloadable .joblib & executive PDF/web report",
    ],
    metricsPreview: [
      { name: "Loop Condition", val: "Gain > 0.5% → Loop" },
      { name: "Research Budget", val: "Iteration 1 / 5" },
    ],
    statePayloadSnippet: {
      decision: "LOOP_TO_PLANNER",
      delta_information_gain: "+0.0184 (> 0.005 threshold)",
      iteration_spent: 1,
      iteration_limit: 5,
      next_target: "node_plan",
    },
    visualComponentType: "loop",
  },
];

/* ── Dynamic LangGraph Neural Background Canvas ─────────────────── */
function LangGraphNeuralCanvas({ activeStageIndex }: { activeStageIndex: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 800);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 450);

    const isMobile = window.innerWidth < 768;
    const nodeCount = isMobile ? 18 : 34;

    interface GraphNode {
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      pulse: number;
      stageCluster: number;
    }

    const nodes: GraphNode[] = Array.from({ length: nodeCount }, (_, i) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.45,
      vy: (Math.random() - 0.5) * 0.45,
      size: Math.random() * 2 + 1,
      pulse: Math.random() * Math.PI,
      stageCluster: i % 6,
    }));

    const resize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };

    window.addEventListener("resize", resize);

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Render neural interconnects
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 110) {
            const alpha = (1 - dist / 110) * 0.18;
            const isHighlighted = nodes[i].stageCluster === activeStageIndex || nodes[j].stageCluster === activeStageIndex;

            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = isHighlighted
              ? `rgba(140,255,32, ${alpha * 2.2})`
              : `rgba(100, 116, 139, ${alpha * 0.7})`;
            ctx.lineWidth = isHighlighted ? 1.2 : 0.75;
            ctx.stroke();

            // Energy packet flow along edge
            if (isHighlighted && (i + j) % 4 === 0) {
              const travel = (Math.sin(nodes[i].pulse * 1.5) + 1) / 2;
              const px = nodes[i].x + (nodes[j].x - nodes[i].x) * travel;
              const py = nodes[i].y + (nodes[j].y - nodes[i].y) * travel;
              ctx.beginPath();
              ctx.arc(px, py, 1.4, 0, Math.PI * 2);
              ctx.fillStyle = "rgba(107, 219, 205, 0.9)";
              ctx.fill();
            }
          }
        }
      }

      // Render node vertices
      for (const node of nodes) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.size, 0, Math.PI * 2);
        const isActiveCluster = node.stageCluster === activeStageIndex;
        ctx.fillStyle = isActiveCluster ? "rgba(140,255,32, 0.85)" : "rgba(148, 163, 184, 0.3)";
        ctx.fill();

        node.x += node.vx;
        node.y += node.vy;
        node.pulse += 0.03;

        if (node.x < 0 || node.x > width) node.vx *= -1;
        if (node.y < 0 || node.y > height) node.vy *= -1;
      }

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", resize);
      if (animId) cancelAnimationFrame(animId);
    };
  }, [activeStageIndex]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 size-full pointer-events-none opacity-40 -z-0"
    />
  );
}

export function WorkflowDiagram() {
  const reducedMotion = useReducedMotion();
  const [active, setActive] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [viewMode, setViewMode] = useState<"visual" | "payload">("visual");

  // Auto-advance through the 6 nodes every 2.8 seconds when playing
  useEffect(() => {
    if (reducedMotion || !isPlaying) return;
    const timer = window.setInterval(() => {
      setActive((curr) => (curr + 1) % STAGES.length);
    }, 2800);
    return () => window.clearInterval(timer);
  }, [reducedMotion, isPlaying]);

  const activeStage = STAGES[active];
  const Icon = activeStage.icon;

  return (
    <section id="workflow" className="scroll-mt-20 border-b border-border-subtle py-24 sm:py-32">
      <div className="mx-auto flex max-w-7xl flex-col gap-12 px-4 sm:px-6">
        
        {/* Section Header */}
        <div className="flex flex-col gap-4 max-w-3xl">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.24em] text-brand-300">
            <Sparkles className="size-3.5" /> Cyclic LangGraph Orchestration
          </div>
          <h2 className="text-balance text-4xl font-semibold tracking-tight text-text sm:text-5xl">
            A stateful research graph that knows what to try next.
          </h2>
          <p className="text-pretty text-lg leading-relaxed text-text-secondary">
            Evidra is not a simple linear script. It runs an autonomous, stateful LangGraph state machine that profiles data, formulates hypotheses, executes 5-fold cross-validation, and iteratively loops until information gains stabilize.
          </p>
        </div>

        {/* Main Interactive Circuit Container */}
        <div className="relative overflow-hidden rounded-2xl border border-border bg-surface-1 shadow-2xl">
          
          {/* Neural Canvas Background */}
          <LangGraphNeuralCanvas activeStageIndex={active} />

          {/* Top Status Telemetry Bar */}
          <div className="relative z-10 flex flex-col gap-3 border-b border-border-subtle bg-surface-2/70 px-6 py-4 backdrop-blur-md sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3 font-mono text-xs text-text-secondary">
              <span className="flex size-2 rounded-full bg-success-400 animate-pulse-soft" />
              <span className="font-semibold text-text">LANGGRAPH THREAD #run_live_01</span>
              <span className="text-text-muted">•</span>
              <span className="text-brand-300 font-mono">{activeStage.nodeName}</span>
            </div>

            {/* Interactive Control Buttons */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="flex items-center gap-1.5 rounded-md border border-border bg-surface-3 px-3 py-1 font-mono text-xs text-text-secondary hover:text-text cursor-pointer transition-colors"
                title={isPlaying ? "Pause automated cycle" : "Resume cycle"}
              >
                {isPlaying ? <Pause className="size-3 text-warning-400" /> : <Play className="size-3 text-success-400" />}
                <span>{isPlaying ? "Pause Loop" : "Resume Loop"}</span>
              </button>

              <span className="rounded-full border border-brand-500/30 bg-brand-500/10 px-3 py-1 font-mono text-xs text-brand-300">
                Stage {activeStage.step}/06
              </span>
            </div>
          </div>

          <div className="relative z-10 p-6 sm:p-10 lg:p-12">
            
            {/* Structured Pipeline Display */}
            <div className="flex flex-col gap-8">
              
              {/* Row 1: Stages 01 -> 02 -> 03 */}
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                    Phase A: Understanding &amp; Execution
                  </span>
                  <span className="font-mono text-[11px] text-brand-300">
                    Step 1 ➔ 2 ➔ 3
                  </span>
                </div>

                <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr_auto_1fr] lg:items-center">
                  
                  {/* Stage 01 */}
                  <StageCard
                    stage={STAGES[0]}
                    isActive={active === 0}
                    onClick={() => { setActive(0); setIsPlaying(false); }}
                  />

                  {/* Fixed Connector 1 -> 2 */}
                  <div className="hidden lg:flex items-center justify-center px-1">
                    <div className="relative flex items-center">
                      <div className="h-0.5 w-8 bg-border-subtle" />
                      <div
                        className={`h-0.5 w-8 transition-colors duration-500 ${
                          active >= 1 ? "bg-brand-400 shadow-[0_0_8px_rgba(140,255,32,0.8)]" : "bg-border-subtle"
                        }`}
                      />
                      <ArrowRight
                        className={`size-4 -ml-1 transition-colors duration-500 ${
                          active >= 1 ? "text-brand-400" : "text-text-muted"
                        }`}
                      />
                    </div>
                  </div>

                  {/* Stage 02 */}
                  <StageCard
                    stage={STAGES[1]}
                    isActive={active === 1}
                    onClick={() => { setActive(1); setIsPlaying(false); }}
                  />

                  {/* Fixed Connector 2 -> 3 */}
                  <div className="hidden lg:flex items-center justify-center px-1">
                    <div className="relative flex items-center">
                      <div className="h-0.5 w-8 bg-border-subtle" />
                      <div
                        className={`h-0.5 w-8 transition-colors duration-500 ${
                          active >= 2 ? "bg-brand-400 shadow-[0_0_8px_rgba(140,255,32,0.8)]" : "bg-border-subtle"
                        }`}
                      />
                      <ArrowRight
                        className={`size-4 -ml-1 transition-colors duration-500 ${
                          active >= 2 ? "text-brand-400" : "text-text-muted"
                        }`}
                      />
                    </div>
                  </div>

                  {/* Stage 03 */}
                  <StageCard
                    stage={STAGES[2]}
                    isActive={active === 2}
                    onClick={() => { setActive(2); setIsPlaying(false); }}
                  />

                </div>
              </div>

              {/* Inter-Row Transition Connector */}
              <div className="hidden lg:flex items-center justify-between px-6 py-1">
                <div className="h-px flex-1 bg-border-subtle" />
                <div className="flex items-center gap-2 rounded-full border border-border bg-surface-2 px-4 py-1.5 font-mono text-[11px] text-text-muted">
                  <span>Out-of-Fold Predictions &amp; Model Weights</span>
                  <ArrowDown className={`size-3.5 ${active >= 3 ? "text-brand-400 animate-bounce" : "text-text-muted"}`} />
                </div>
                <div className="h-px flex-1 bg-border-subtle" />
              </div>

              {/* Row 2: Stages 04 -> 05 -> 06 */}
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                    Phase B: Evaluation, Memory &amp; Adaptive Routing
                  </span>
                  <span className="font-mono text-[11px] text-brand-300">
                    Step 4 ➔ 5 ➔ 6
                  </span>
                </div>

                <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr_auto_1fr] lg:items-center">
                  
                  {/* Stage 04 */}
                  <StageCard
                    stage={STAGES[3]}
                    isActive={active === 3}
                    onClick={() => { setActive(3); setIsPlaying(false); }}
                  />

                  {/* Fixed Connector 4 -> 5 */}
                  <div className="hidden lg:flex items-center justify-center px-1">
                    <div className="relative flex items-center">
                      <div className="h-0.5 w-8 bg-border-subtle" />
                      <div
                        className={`h-0.5 w-8 transition-colors duration-500 ${
                          active >= 4 ? "bg-brand-400 shadow-[0_0_8px_rgba(140,255,32,0.8)]" : "bg-border-subtle"
                        }`}
                      />
                      <ArrowRight
                        className={`size-4 -ml-1 transition-colors duration-500 ${
                          active >= 4 ? "text-brand-400" : "text-text-muted"
                        }`}
                      />
                    </div>
                  </div>

                  {/* Stage 05 */}
                  <StageCard
                    stage={STAGES[4]}
                    isActive={active === 4}
                    onClick={() => { setActive(4); setIsPlaying(false); }}
                  />

                  {/* Fixed Connector 5 -> 6 */}
                  <div className="hidden lg:flex items-center justify-center px-1">
                    <div className="relative flex items-center">
                      <div className="h-0.5 w-8 bg-border-subtle" />
                      <div
                        className={`h-0.5 w-8 transition-colors duration-500 ${
                          active >= 5 ? "bg-brand-400 shadow-[0_0_8px_rgba(140,255,32,0.8)]" : "bg-border-subtle"
                        }`}
                      />
                      <ArrowRight
                        className={`size-4 -ml-1 transition-colors duration-500 ${
                          active >= 5 ? "text-brand-400" : "text-text-muted"
                        }`}
                      />
                    </div>
                  </div>

                  {/* Stage 06 */}
                  <StageCard
                    stage={STAGES[5]}
                    isActive={active === 5}
                    onClick={() => { setActive(5); setIsPlaying(false); }}
                  />

                </div>
              </div>

              {/* Loop-back Feedback Ribbon */}
              <div className="rounded-xl border border-brand-500/25 bg-brand-500/[0.05] p-4 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono">
                <div className="flex items-center gap-2.5 text-text">
                  <span className="flex size-6 items-center justify-center rounded-md bg-brand-500/20 text-brand-300">
                    <RotateCcw className="size-3.5 animate-spin-slow" />
                  </span>
                  <span className="font-semibold text-brand-300">Adaptive Feedback Routing:</span>
                  <span className="text-text-secondary">If Information Gain &gt; 0.5% &amp; budget remains ➔ injects memory &amp; loops back to Stage 02 (Planner).</span>
                </div>
                <span className="rounded-full border border-brand-500/30 bg-surface-2 px-3 py-1 text-text-muted shrink-0">
                  Max 5 Iterations Budget
                </span>
              </div>

            </div>

            {/* Deep Stage Inspector Box & Live State Machine View */}
            <div className="mt-8 overflow-hidden rounded-2xl border border-border bg-surface-2/90 p-5 sm:p-7 backdrop-blur-md shadow-xl">
              
              {/* Header with Switcher Tabs */}
              <div className="flex flex-col gap-4 border-b border-border-subtle pb-5 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3">
                  <span className="flex size-9 items-center justify-center rounded-lg border border-brand-500/40 bg-brand-500/20 text-brand-300 shadow-sm">
                    <Icon className="size-4.5" />
                  </span>
                  <div>
                    <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
                      Active LangGraph Node Inspection
                    </span>
                    <h4 className="text-lg font-bold text-text">
                      {activeStage.label} <span className="font-mono text-xs font-normal text-brand-300">({activeStage.nodeName})</span>
                    </h4>
                  </div>
                </div>

                {/* View Switcher Toggle */}
                <div className="flex items-center rounded-lg border border-border bg-surface-3 p-1 font-mono text-xs">
                  <button
                    onClick={() => setViewMode("visual")}
                    className={`flex items-center gap-1.5 rounded-md px-3 py-1 font-medium transition-colors cursor-pointer ${
                      viewMode === "visual" ? "bg-brand-500 text-bg shadow-sm" : "text-text-muted hover:text-text"
                    }`}
                  >
                    <Activity className="size-3.5" /> Visual Flow
                  </button>
                  <button
                    onClick={() => setViewMode("payload")}
                    className={`flex items-center gap-1.5 rounded-md px-3 py-1 font-medium transition-colors cursor-pointer ${
                      viewMode === "payload" ? "bg-brand-500 text-bg shadow-sm" : "text-text-muted hover:text-text"
                    }`}
                  >
                    <Code2 className="size-3.5" /> Immutable State
                  </button>
                </div>
              </div>

              {/* Dynamic View Body */}
              <div className="py-5">
                <AnimatePresence mode="wait">
                  {viewMode === "visual" ? (
                    <motion.div
                      key={`visual-${activeStage.id}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.25 }}
                      className="grid gap-6 lg:grid-cols-[1.2fr_.8fr] lg:items-center"
                    >
                      {/* Left: Core Node Action & Outputs */}
                      <div className="flex flex-col gap-4">
                        <p className="text-sm leading-relaxed text-text-secondary">
                          {activeStage.tagline}
                        </p>
                        <div className="flex flex-col gap-2">
                          <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                            Node State Transformations:
                          </span>
                          <ul className="grid gap-2 text-xs text-text-secondary">
                            {activeStage.coreOutputs.map((out, i) => (
                              <li key={i} className="flex items-start gap-2 rounded-lg border border-border-subtle bg-surface-3/60 p-2.5">
                                <ShieldCheck className="size-4 text-brand-300 shrink-0 mt-0.5" />
                                <span>{out}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Right: Stage Live Metric Cards */}
                      <div className="flex flex-col gap-3 rounded-xl border border-border-subtle bg-surface-3/80 p-4">
                        <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                          Telemetry &amp; Decision Criteria:
                        </span>
                        <div className="grid gap-2">
                          {activeStage.metricsPreview.map((met) => (
                            <div
                              key={met.name}
                              className="flex items-center justify-between rounded-lg border border-border-subtle bg-surface-2 p-3 font-mono text-xs"
                            >
                              <span className="text-text-muted">{met.name}</span>
                              <span className="font-bold text-brand-300">{met.val}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key={`payload-${activeStage.id}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.25 }}
                      className="rounded-xl border border-border-subtle bg-surface-1 p-4 font-mono text-xs text-text-secondary overflow-x-auto shadow-inner"
                    >
                      <div className="flex items-center justify-between border-b border-border-subtle pb-2 mb-3 text-[11px] text-text-muted">
                        <span>StateGraph Thread Memory Key: <span className="text-brand-300 font-semibold">{activeStage.id}</span></span>
                        <span className="text-success-400">● Validated by Pydantic v2</span>
                      </div>
                      <pre className="text-text-muted text-[11px] leading-relaxed">
                        {JSON.stringify(activeStage.statePayloadSnippet, null, 2)}
                      </pre>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Bottom Step Switcher Dots */}
              <div className="flex items-center justify-between border-t border-border-subtle pt-4 text-xs">
                <div className="flex items-center gap-2 text-text-muted">
                  <RotateCcw className="size-3.5 text-brand-400 animate-spin-slow" />
                  <span>Click any stage card to step through the state machine</span>
                </div>
                <div className="flex gap-1.5" aria-label="Select workflow stage">
                  {STAGES.map((st, i) => (
                    <button
                      key={st.id}
                      onClick={() => { setActive(i); setIsPlaying(false); }}
                      className={`h-1.5 rounded-full transition-all cursor-pointer ${
                        active === i ? "w-8 bg-brand-400" : "w-2.5 bg-surface-4 hover:bg-text-muted"
                      }`}
                      aria-label={`Select stage ${st.step}`}
                    />
                  ))}
                </div>
              </div>
            </div>

          </div>

          {/* Bottom Footnote Cards */}
          <div className="relative z-10 grid border-t border-border-subtle sm:grid-cols-3 bg-surface-2/50 text-xs backdrop-blur-md">
            <div className="border-b border-border-subtle px-6 py-4 font-mono text-text-muted last:border-b-0 sm:border-b-0 sm:border-r">
              <span className="text-text font-semibold">SemanticProfile</span> ➔ Immutable State Graph
            </div>
            <div className="border-b border-border-subtle px-6 py-4 font-mono text-text-muted last:border-b-0 sm:border-b-0 sm:border-r">
              <span className="text-text font-semibold">5-Fold Leakage Shield</span> ➔ Out-of-fold validation
            </div>
            <div className="px-6 py-4 font-mono text-text-muted">
              <span className="text-text font-semibold">Self-Contained Export</span> ➔ .joblib + Report
            </div>
          </div>

        </div>

      </div>
    </section>
  );
}

function StageCard({
  stage,
  isActive,
  onClick,
}: {
  stage: StageInfo;
  isActive: boolean;
  onClick: () => void;
}) {
  const StageIcon = stage.icon;

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={isActive}
      className={`
        group relative flex flex-col items-start gap-3.5 rounded-xl border p-5 text-left transition-all cursor-pointer select-none w-full
        ${
          isActive
            ? "border-brand-500 bg-surface-2/95 shadow-xl shadow-brand-500/10 ring-1 ring-brand-500/40 backdrop-blur-md"
            : "border-border-subtle bg-surface-2/60 hover:border-border hover:bg-surface-2/90"
        }
      `}
    >
      {/* Top row: step number + icon */}
      <div className="flex w-full items-center justify-between">
        <span
          className={`flex size-10 items-center justify-center rounded-lg border transition-colors ${
            isActive
              ? "border-brand-500/60 bg-brand-500/20 text-brand-300"
              : "border-border bg-surface-3 text-text-muted group-hover:text-text"
          }`}
        >
          <StageIcon className="size-5" aria-hidden="true" />
        </span>
        <span className="font-mono text-xs font-semibold text-text-muted">
          STAGE {stage.step}
        </span>
      </div>

      {/* Label & Tagline */}
      <div className="flex flex-col gap-1 min-w-0">
        <h3 className="text-base font-bold tracking-tight text-text">
          {stage.label}
        </h3>
        <p className="text-xs leading-relaxed text-text-muted line-clamp-2">
          {stage.tagline}
        </p>
      </div>

      {/* Active indicator badge */}
      {isActive && (
        <div className="mt-1 inline-flex items-center gap-1.5 font-mono text-[11px] font-medium text-brand-300">
          <Zap className="size-3" /> Active Orchestration Node
        </div>
      )}
    </button>
  );
}



