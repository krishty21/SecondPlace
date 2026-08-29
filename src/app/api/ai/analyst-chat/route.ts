import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const SYSTEM_PROMPT = `You are the CipherMind Sentinel Analyst Copilot — a SOC assistant answering analyst questions about network intrusion detections.
The underlying system was trained on the UNSW-NB15 dataset: network FLOW records only (protocol, service, state, packet/byte counts, rates, TTLs, loads, jitter, TCP timing, connection counts).

RULES:
- Ground every claim in the provided CONTEXT (incident evidence, model outputs, metrics). Cite the numbers.
- NEVER invent IPs, hostnames, users, malware families, CVEs, or timestamps as real. Entities are simulated pseudo-entities; timestamps are simulated replay offsets. If asked about real IPs/users, explain those fields do not exist in this dataset.
- Be concise (<= 200 words unless asked for detail), use security-operations vocabulary.
- If uncertainty exists (rare classes, borderline confidence), say so explicitly.
- Recommendations are suggestions for the analyst, not confirmed facts.`;

function fallbackReply(question: string): string {
  const q = question.toLowerCase();
  if (q.includes("why") || q.includes("explain") || q.includes("reason")) {
    return "Model reasoning is available in the Explainability tab: each detection lists the top contributing features (exact TreeSHAP where precomputed, otherwise Saabas path attributions computed live), the calibrated attack probability, and the anomaly score. The explanation only references flow-level features present in UNSW-NB15 — no IP or user fields exist in this dataset. (Deterministic fallback reply — LLM unavailable.)";
  }
  if (q.includes("risk") || q.includes("priority") || q.includes("score")) {
    return "Risk scores combine: calibrated attack confidence (32%), anomaly score (18%), category severity (20%), class rarity (8%), prediction uncertainty (10%), and correlation boost from related alerts (12%). The formula is transparent and configurable — see docs/THREAT_SCORING.md. Bands: 0-24 Low, 25-49 Medium, 50-74 High, 75-100 Critical. (Deterministic fallback reply — LLM unavailable.)";
  }
  if (q.includes("incident") || q.includes("correlat")) {
    return "Alerts are correlated into incidents using predicted-category similarity, temporal proximity within a sliding 3-minute window (simulated time), behavioral distance on standardized flow features, and simulated pseudo-entity overlap. Every incident consolidates many alerts into one prioritized story with an attack timeline. (Deterministic fallback reply — LLM unavailable.)";
  }
  if (q.includes("data") || q.includes("dataset") || q.includes("unsw")) {
    return "The models were trained on the official UNSW-NB15 training set (175,341 flows) and evaluated exactly once on the official test set (82,332 flows). The supplied 45-column CSVs contain flow statistics only — no source/destination IPs, ports-as-addresses, users, devices, or capture timestamps. Any IP-like or timeline metadata in this UI is clearly-labeled simulation for demo purposes. (Deterministic fallback reply — LLM unavailable.)";
  }
  return "I can help interpret detections, incident correlations, risk scores, model explanations, and evaluation metrics. Ask about a specific incident, a feature contribution, or the model's evaluation. Note: this is the deterministic fallback (LLM unavailable), so answers are limited to general system documentation.";
}

export async function POST(request: Request) {
  let messages: { role: string; content: string }[] = [];
  let context: Record<string, unknown> = {};
  try {
    const body = (await request.json()) as {
      messages?: { role: string; content: string }[];
      context?: Record<string, unknown>;
    };
    messages = (body.messages ?? []).filter((m) => typeof m.content === "string" && m.content.length > 0).slice(-10);
    context = body.context ?? {};
    if (!messages.length) {
      return NextResponse.json({ error: "messages required" }, { status: 400 });
    }
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }

  const lastUser = [...messages].reverse().find((m) => m.role === "user")?.content ?? "";

  try {
    const { default: ZAI } = await import("z-ai-web-dev-sdk");
    const zai = await ZAI.create();
    const contextStr = Object.keys(context).length
      ? `\n\nCONTEXT (structured evidence, ground your answer in it):\n${JSON.stringify(context).slice(0, 6000)}`
      : "";
    const chatMessages: { role: "user" | "assistant" | "system"; content: string }[] = [
      { role: "assistant", content: SYSTEM_PROMPT },
      ...messages.map((m) => ({
        role: (m.role === "assistant" ? "assistant" : "user") as "user" | "assistant",
        content: m.content.slice(0, 4000),
      })),
    ];
    // inject context into the last user message
    const lastIdx = chatMessages.length - 1;
    chatMessages[lastIdx] = { role: "user", content: chatMessages[lastIdx].content + contextStr };
    const completion = await zai.chat.completions.create({
      messages: chatMessages,
      thinking: { type: "disabled" },
    });
    const reply = completion.choices[0]?.message?.content;
    if (!reply) throw new Error("empty LLM response");
    return NextResponse.json({ reply, source: "llm" });
  } catch (e) {
    console.error("[ai/analyst-chat] LLM failed, using fallback:", (e as Error).message);
    return NextResponse.json({ reply: fallbackReply(lastUser), source: "fallback" });
  }
}
