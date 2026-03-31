import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ─── Helpers ──────────────────────────────────────────────────────────────────

function parseJSON(raw, fallback) {
  try {
    return JSON.parse(raw.trim());
  } catch {
    console.error("JSON parse failed. Raw output:", raw);
    return fallback;
  }
}

/**
 * Removes duplicate and near-duplicate entries from a whatToDo array.
 * Uses Jaccard similarity on word tokens — threshold 0.6.
 * Caps result at 3 items.
 */
function dedupeWhatToDo(items) {
  if (!Array.isArray(items) || items.length === 0) return items;

  const normalize = (s) =>
    String(s)
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .replace(/\s+/g, " ")
      .trim();

  // Words longer than 2 chars carry meaning; shorter ones are stopwords
  const tokenize = (s) => normalize(s).split(" ").filter((w) => w.length > 2);

  const jaccard = (a, b) => {
    const setA = new Set(tokenize(a));
    const setB = new Set(tokenize(b));
    if (setA.size === 0 && setB.size === 0) return 1;
    const intersection = [...setA].filter((w) => setB.has(w)).length;
    const union = new Set([...setA, ...setB]).size;
    return intersection / union;
  };

  const kept = [];
  for (const item of items) {
    if (typeof item !== "string" || !item.trim()) continue;
    const isDupe = kept.some((k) => jaccard(k, item) >= 0.6);
    if (!isDupe) kept.push(item);
    if (kept.length >= 3) break;
  }

  return kept;
}

// ─── Shared prompts ───────────────────────────────────────────────────────────

const replySystemPrompt = `CRITICAL — ACTION EXECUTION ONLY

You are NOT a conversational assistant.
You are executing a fixed action.

Input:
- action (locked)
- language

You MUST:
- generate a reply that DIRECTLY executes the action
- NOT ask questions unless the action explicitly requires it
- NOT explore or continue the conversation
- NOT infer new intent

────────────────────────────────────────────────────────
ACTION RULES
────────────────────────────────────────────────────────

If action = USE_GRAB or DECLINE:
→ reply MUST be a decline or redirect
→ NEVER ask questions

If action = ASK_CONTRACT:
→ reply asks for contract ONLY

If action = EXIT:
→ reply ends the interaction

If the output does not match the action → regenerate until it does.

────────────────────────────────────────────────────────

You are Kova's reply generator — Phase 2 of a strict two-phase pipeline.

You receive a LOCKED action directive. Your ONLY job is to execute it.

────────────────────────────────────────────────────────
CRITICAL RULES (non-negotiable)
────────────────────────────────────────────────────────

1. EXECUTE THE ACTION — do not re-evaluate it or change it.
2. DO NOT re-assess risk or intent.
3. DO NOT reference or consider the original message — you do not have it.
4. DO NOT introduce new intent beyond what the action requires.
5. DO NOT follow curiosity, explore further, or soften the decision.
6. ONE reply only. 1 sentence preferred. 2 maximum.

────────────────────────────────────────────────────────
ACTION → REPLY EXECUTION
────────────────────────────────────────────────────────

USE_GRAB
→ Decline off-app transport. Use official app.
→ Vietnamese: "Anh đi Grab, cảm ơn."
→ English: "I'll just use Grab, thanks."

ASK_CONTRACT
→ Request the contract before proceeding. One ask only.
→ Vietnamese: "Anh cần xem hợp đồng trước nhé."
→ English: "I need to see the contract first."

DECLINE
→ Reject cleanly. No explanation needed.
→ Vietnamese: "Thôi cảm ơn em, anh không cần."
→ English: "Thanks, but I'll pass."

EXIT
→ End the interaction. Minimal.
→ Vietnamese: "Anh bận rồi nhé."
→ English: "I've got to go."

HOLD_PAYMENT
→ Do not agree to send money. Express delay or refusal.
→ Vietnamese: "Chưa chuyển được — anh xem lại đã."
→ English: "I can't send that right now."

VERIFY_IDENTITY
→ Request proof or confirmation. Calm, one request.
→ Vietnamese: "Em có thể xác nhận thêm cho anh không?"
→ English: "Can you confirm that for me?"

HOLD_POSITION
→ Stay firm. No softening.
→ Vietnamese: "Anh chưa đồng ý mức đó."
→ English: "That doesn't work for me."

MATCH_ENERGY
→ Use the context field to generate a natural, human reply.
→ Match the energy of the conversation — no agenda, no strategy.
→ One sentence, instinctive, sounds like a real person.

IGNORE
→ Output empty strings. No reply.

────────────────────────────────────────────────────────
LANGUAGE + CULTURAL RULES
────────────────────────────────────────────────────────

Reply in the language specified. Match cultural register.

Vietnamese:
- Casual particles: "nhé", "đi", "vậy", "nè"
- User = Anh (I/me). Other person = Em or Bạn. Never mix.
- "Anh cần..." / "Anh sẽ..." = user's own statement — NOT a command.
- NEVER produce commands directed at the user.

English:
- Contractions fine. Incomplete sentences fine.
- Brief and direct.

Other languages:
- Match rhythm, register, and directness of that language.
- Never produce a correct-but-foreign sentence.

────────────────────────────────────────────────────────
REALISM CHECK (mandatory before output)
────────────────────────────────────────────────────────

"Would a real person type this exact sentence in a chat?"
If not → simplify, shorten, make it natural.
Output ONLY the version that passes.

────────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────────

Return ONLY a valid JSON object — no markdown, no extra text:
{
  "native": "Reply in the conversation's language.",
  "english": "Translate from the USER's perspective. Boundaries use 'I'. Requests use 'you'. Never a command to the user."
}`;

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get("/", (req, res) => {
  res.send("Kova backend is running");
});

// ── /analyze — understand the situation ──────────────────────────────────────

app.post("/analyze", async (req, res) => {
  try {
    const { image, selectedMessageImage, croppedImage, selectedMessage, tapX, tapY } = req.body;

    const contextPrefix = selectedMessageImage
      ? `You are given TWO images:\n1. SELECTED MESSAGE — the exact region the user highlighted. Analyze only this.\n2. FULL SCREENSHOT — background context for tone and relationship only.`
      : selectedMessage
      ? `The user selected this exact message: "${selectedMessage}". Analyze only this message.\nThe full screenshot is background context only.`
      : croppedImage
        ? `You are given TWO images:\n1. SELECTED MESSAGE — the exact message the user tapped. Analyze only this.\n2. FULL CONVERSATION — background context for tone and relationship only.`
        : `The user tapped the message at approximately ${tapX}% from left and ${tapY}% from top.\nAnalyze only that message. Everything else is background context.`;

    const schema = `Analyze the selected message and return ONLY a valid JSON object — no markdown, no extra text:

{
  "riskLevel": "Low" | "Medium" | "High",
  "meaning": "1-2 lines. What is actually happening and why it matters. No filler.",
  "recommendedAction": "One of: USE_GRAB | DECLINE | EXIT | ASK_CONTRACT | HOLD_PAYMENT | HOLD_POSITION | VERIFY_IDENTITY | MATCH_ENERGY | IGNORE"
}

Risk guidelines:
- Low: normal conversation, no pressure or money involved
- Medium: unclear intent, possible manipulation, informal financial offers
- High: active scam signals, urgent money pressure, off-platform requests

Pick recommendedAction based on what genuinely fits the situation.`;

    const userContent = [];
    if (selectedMessageImage) {
      userContent.push({ type: "text", text: "Image 1: selected message. Image 2: full screenshot for context." });
      userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${selectedMessageImage}` } });
    } else if (croppedImage) {
      userContent.push({ type: "text", text: "Selected message (image 1). Full conversation (image 2)." });
      userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${croppedImage}` } });
    } else {
      userContent.push({ type: "text", text: "Analyse this screenshot." });
    }
    userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${image}` } });

    const analysisResponse = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: contextPrefix + "\n\n" + schema },
        { role: "user",   content: userContent },
      ],
    });

    const raw = analysisResponse.choices[0].message.content;
    const parsed = parseJSON(raw, {
      riskLevel: "Low",
      meaning: "Could not read the image.",
      recommendedAction: "DECLINE",
    });

    if (!["Low", "Medium", "High"].includes(parsed.riskLevel)) parsed.riskLevel = "Low";
    if (typeof parsed.meaning !== "string" || !parsed.meaning.trim()) parsed.meaning = "";
    if (typeof parsed.recommendedAction !== "string") parsed.recommendedAction = "DECLINE";
    parsed.recommendedAction = parsed.recommendedAction.toUpperCase().replace(/\s+/g, "_");

    res.json(parsed);

  } catch (err) {
    console.error("/analyze error:", err.message);
    res.json({
      riskLevel: "Low",
      meaning: "Backend error — please try again.",
      recommendedAction: "DECLINE",
    });
  }
});

// ── /respond — user-driven reply generation ───────────────────────────────────

app.post("/respond", async (req, res) => {
  try {
    const { userIntent, language, context } = req.body;
    const { riskLevel = "Low", meaning = "", recommendedAction = "" } = context || {};

    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: `You generate 2–3 reply options for a user based on what they want to say.

Rules:
- Each reply is 1 sentence max
- Sound like a real local speaker — not a translation
- Interpret the user's intent naturally; do NOT translate their words literally
- Match the tone and register of the target language
- If riskLevel is High or Medium: replies should feel controlled and cautious, but still follow the user's intent

Language notes:
- Vietnamese: use casual particles (nhé, đi, vậy), Anh/Em pronouns, short sentences
- English: contractions and incomplete sentences are fine
- Other languages: match the natural register and rhythm

Return ONLY a valid JSON object — no markdown, no extra text:
{ "replies": ["option 1", "option 2", "option 3"] }`,
        },
        {
          role: "user",
          content: `User intent: "${userIntent}"\nLanguage: ${language}\nSituation: ${meaning}\nRisk: ${riskLevel}\nRecommended action: ${recommendedAction}`,
        },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, { replies: [] });
    if (!Array.isArray(parsed.replies)) parsed.replies = [];
    parsed.replies = parsed.replies.filter((r) => typeof r === "string" && r.trim());
    res.json(parsed);

  } catch (err) {
    console.error("/respond error:", err.message);
    res.json({ replies: [] });
  }
});

// ── /ocr — extract text from a cropped image ──────────────────────────────────

app.post("/ocr", async (req, res) => {
  try {
    const { image } = req.body;
    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      max_tokens: 300,
      messages: [
        {
          role: "system",
          content: `Extract the exact text content from this chat message image. Return ONLY the raw message text as it appears — nothing else. No quotes, no labels, no explanation. If multiple messages are visible, extract only the most prominent/central one.`,
        },
        {
          role: "user",
          content: [{ type: "image_url", image_url: { url: `data:image/jpeg;base64,${image}` } }],
        },
      ],
    });
    const text = response.choices[0].message.content?.trim() ?? "";
    res.json({ text });
  } catch (err) {
    console.error("/ocr error:", err.message);
    res.json({ text: "" });
  }
});

// ── /decide — Phase 1: decision engine ────────────────────────────────────────

app.post("/decide", async (req, res) => {
  try {
    const { selectedMessage, conversationContext } = req.body;

    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: `CRITICAL — ACTION SELECTION OVERRIDE

The action MUST be based on real-world behavior, not user intent.

────────────────────────────────────────────────────────
RULES
────────────────────────────────────────────────────────

If the scenario is risky (scam patterns, off-platform offers, money risk):
→ IGNORE user curiosity or requests to explore
→ DO NOT choose actions like ASK, EXPLORE, CLARIFY

Instead, ALWAYS choose a safe action:
- off-app taxi or transport offer → USE_GRAB
- unknown or informal payment → DECLINE
- suspicious offer or pressure → EXIT

User input like "I want to know more" / "ask them" / "find out"
MUST NOT influence action selection in risky scenarios.

FINAL RULE:
Action must reflect what a real person SHOULD do — not what the user WANTS to do.

────────────────────────────────────────────────────────

You are Kova's decision engine — Phase 1 of a strict two-phase pipeline.

Your ONLY job: analyze the situation and output a LOCKED decision.
You are NOT generating a reply. You are deciding what action to take.
This output becomes the single source of truth. The reply generator will execute it — it will NOT re-evaluate.

────────────────────────────────────────────────────────
DECISION PROCESS
────────────────────────────────────────────────────────

STEP 1 — READ THE SITUATION
Understand what is actually happening — not just the surface words.
Identify who is speaking, what they want, and any signals (pressure, money, urgency, tone).

STEP 2 — CLASSIFY RISK

LOW  — Normal conversation, casual request, no pressure or money involved.
MEDIUM — Unclear intent, possible manipulation, informal financial offers.
HIGH  — Active scam signals, urgent money pressure, off-platform requests, clear threat.

HARD MINIMUMS (non-overridable):
- Off-app taxi / transport offer → HIGH
- Payment before contract → HIGH
- Urgency + money combined → HIGH
- Informal money exchange ("better rate", "no fee", "I can help exchange") → MEDIUM minimum
- Unsolicited financial help → MEDIUM minimum

Do NOT soften risk because the tone is friendly. Friendly tone does not reduce financial risk.

STEP 3 — PICK PRIMARY GOAL (exactly one)

AVOID   — Scam signals, pressure, discomfort, unsolicited offers, street situations.
VERIFY  — Formal context (landlord, employer, contract) — something is unclear or unconfirmed.
NEGOTIATE — Price, terms, or conditions — user holds position or pushes back.
SOCIAL  — Casual chat, banter, flirting, friendly exchange.

Selection rules:
- AVOID beats VERIFY when scam signals are present.
- LOW-TRUST street situations → AVOID, not VERIFY (no verification is realistic on the street).
- SOCIAL → never apply strategy or caution.

STEP 4 — CHOOSE ONE ACTION

Standard action identifiers:
USE_GRAB        — decline off-app transport, use official rideshare app
DECLINE         — reject offer or request cleanly
ASK_CONTRACT    — request formal paperwork before proceeding
EXIT            — leave the interaction entirely
VERIFY_IDENTITY — request identification or confirmation
HOLD_PAYMENT    — do not send money yet
HOLD_POSITION   — negotiation — stay firm on current terms
MATCH_ENERGY    — social/flirt context — reply naturally and humanly
IGNORE          — do not engage

Use a short descriptive action if none fit exactly. Keep it concise.

SITUATION → ACTION EXAMPLES:
- Taxi driver asks to pay cash off-app → USE_GRAB
- Landlord asks for deposit before contract → ASK_CONTRACT
- Someone pressuring urgently + money → DECLINE
- Casual flirty message → MATCH_ENERGY
- Negotiation, they push too high → HOLD_POSITION
- Suspicious stranger approaching → EXIT

CRITICAL:
If risk is HIGH: action MUST be EXIT, DECLINE, or USE_GRAB — never explore or engage further.
If risk is MEDIUM: action must be protective — never pure engagement.
User curiosity ("ask", "tell me more") does NOT change the action in risky situations.

STEP 5 — ENCODE CONTEXT (required for SOCIAL, optional for others)

For SOCIAL / MATCH_ENERGY: describe the conversational intent in 1 short line.
This is what the reply generator uses to produce a natural human message.

Examples:
"Playful biting comment — match the tease, keep it light."
"Checking in warmly — respond in kind, no agenda."
"Flirty challenge — punchy, slightly charged reply."

For non-SOCIAL actions: set context to empty string "".

────────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────────

Return ONLY a valid JSON object — no markdown, no extra text:
{
  "riskLevel": "LOW" | "MEDIUM" | "HIGH",
  "primaryGoal": "AVOID" | "VERIFY" | "NEGOTIATE" | "SOCIAL",
  "action": "USE_GRAB" | "DECLINE" | "ASK_CONTRACT" | "EXIT" | "MATCH_ENERGY" | ...,
  "context": "Short line for SOCIAL — conversational intent. Empty string otherwise."
}`,
        },
        {
          role: "user",
          content: conversationContext
            ? `Message: "${selectedMessage}"\n\nConversation context:\n${conversationContext}`
            : `Message: "${selectedMessage}"`,
        },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, {
      riskLevel: "LOW",
      primaryGoal: "SOCIAL",
      action: "MATCH_ENERGY",
      context: "",
    });

    if (!["LOW", "MEDIUM", "HIGH"].includes(parsed.riskLevel)) parsed.riskLevel = "LOW";
    if (!["AVOID", "VERIFY", "NEGOTIATE", "SOCIAL"].includes(parsed.primaryGoal)) parsed.primaryGoal = "SOCIAL";
    if (typeof parsed.action !== "string" || !parsed.action.trim()) parsed.action = "MATCH_ENERGY";
    if (typeof parsed.context !== "string") parsed.context = "";

    // Normalize action to strict enum
    const ALLOWED_ACTIONS = ["USE_GRAB", "DECLINE", "EXIT", "ASK_CONTRACT", "PROCEED_WITH_CAUTION", "MATCH_ENERGY", "HOLD_PAYMENT", "HOLD_POSITION", "VERIFY_IDENTITY", "IGNORE"];
    const ACTION_MAP = {
      "use grab": "USE_GRAB", "grab": "USE_GRAB", "use_grab": "USE_GRAB",
      "decline": "DECLINE", "reject": "DECLINE", "refuse": "DECLINE",
      "exit": "EXIT", "leave": "EXIT", "end": "EXIT",
      "ask contract": "ASK_CONTRACT", "ask_contract": "ASK_CONTRACT", "request contract": "ASK_CONTRACT",
      "proceed with caution": "PROCEED_WITH_CAUTION", "proceed_with_caution": "PROCEED_WITH_CAUTION", "caution": "PROCEED_WITH_CAUTION",
      "match energy": "MATCH_ENERGY", "match_energy": "MATCH_ENERGY",
      "hold payment": "HOLD_PAYMENT", "hold_payment": "HOLD_PAYMENT",
      "hold position": "HOLD_POSITION", "hold_position": "HOLD_POSITION",
      "verify identity": "VERIFY_IDENTITY", "verify_identity": "VERIFY_IDENTITY",
      "ignore": "IGNORE",
    };
    const rawAction = parsed.action.trim().toLowerCase();
    const upper = parsed.action.trim().toUpperCase();
    if (ALLOWED_ACTIONS.includes(upper)) {
      parsed.action = upper;
    } else if (ACTION_MAP[rawAction]) {
      parsed.action = ACTION_MAP[rawAction];
    } else {
      parsed.action = "DECLINE"; // safe fallback for any unrecognized free text
    }

    res.json(parsed);

  } catch (err) {
    console.error("/decide error:", err.message);
    res.json({ riskLevel: "LOW", primaryGoal: "SOCIAL", action: "MATCH_ENERGY", context: "" });
  }
});

// ── /reply — Phase 2: execute locked action ────────────────────────────────────

app.post("/reply", async (req, res) => {
  try {
    const { action, language } = req.body;

    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: replySystemPrompt },
        { role: "user", content: `Action: ${action}\nLanguage: ${language}` },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, { native: "", english: "" });
    if (typeof parsed.native !== "string") parsed.native = "";
    if (typeof parsed.english !== "string") parsed.english = "";
    res.json(parsed);

  } catch (err) {
    console.error("/reply error:", err.message);
    res.json({ native: "", english: "" });
  }
});

// ── /refine — refine a generated reply ────────────────────────────────────────

app.post("/refine", async (req, res) => {
  try {
    const { native, instruction, action } = req.body;

    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: `You are Kova. Your job is to generate what the user SHOULD say — not translate what they typed.

────────────────────────────────────────────────────────
ACTION LOCK (CRITICAL — check first)
────────────────────────────────────────────────────────

${action ? `The decision has been locked: action = "${action}"

This action CANNOT be changed by refinement.
Refinement can ONLY adjust: tone, wording, length, delivery.

Examples of what IS allowed:
- "be more firm" → same action, harder delivery
- "shorter" → same action, fewer words
- "be more polite" → same action, warmer tone

Examples of what is NOT allowed:
- "ask them how it works" when action = USE_GRAB → NOT allowed (would change action)
  Exception: if user explicitly insists and it is not extremely dangerous → allow controlled deviation (see REFINEMENT CONTROL RULE below)

RULE: If user instruction would change the action and user is NOT explicitly insisting → adjust tone, keep action.
` : `No locked action provided. Treat the original reply's direction as the default behavior to preserve.
`}
────────────────────────────────────────────────────────
CRITICAL RULE: INPUT IS INTENT, NOT CONTENT
────────────────────────────────────────────────────────

The user's input is an INTENT SIGNAL — what they want to achieve.
It is NOT the message to send.
Do NOT translate it. Do NOT reuse their wording. Do NOT mirror their sentence structure.

PROCESS:
1. Read the original reply (what Kova already suggested)
2. Read the user's input — interpret it as INTENT
3. Generate a NEW reply that achieves that intent, within the bounds of the locked action

INTENT EXAMPLES:

"I want to know more about this, it might be good"
→ Intent: get more info without committing
→ Generate: "Cái này cụ thể như thế nào vậy?" / "Tell me more — what's involved exactly?"
→ NOT: a translation of "I want to know more"

"I need it before 1pm"
→ Intent: set a time condition
→ Generate: "Anh cần trước 1 giờ nhé." / "Need it before 1pm."
→ NOT: "I need it before 1:00PM"

"maybe explore this more"
→ Intent: show interest without committing
→ Generate: "Nghe có vẻ được — anh muốn hiểu rõ hơn trước." / "Sounds interesting — can you walk me through it?"

"I don't want to be rude but I want to decline"
→ Intent: soft decline
→ Generate: "Cảm ơn em nhé, lần này anh chưa tiện." / "Thanks, but I'll pass this time."

────────────────────────────────────────────────────────
REFINEMENT CONTROL RULE
────────────────────────────────────────────────────────

Default: follow the safest, most protective decision established in the original reply.
If the user explicitly insists on different behavior, and it is NOT extremely dangerous: allow controlled deviation.

Priority:
1. Default = follow safest decision (the original reply's direction)
2. If user pushes for alternative behavior → allow it, but keep it realistic and aware
3. Never blindly follow unsafe curiosity — guide first, allow second

Example:
Original reply default: "I'll use Grab, thanks." (protective)
User insists: "ask them how it works"
→ Allowed: "Xe anh chạy kiểu nào vậy?" (curious but controlled — not naive)
→ NOT: "Yes great idea, please explain everything to me!"

Rules:
- Do not block user control entirely — they may have good reasons to deviate
- Do not ignore safety signals — show awareness even when complying
- Extreme danger (scam confirmation, dangerous payment, clear threat) = do not allow override
- Moderate situations = allow deviation with realism preserved

────────────────────────────────────────────────────────
SHORTHAND INSTRUCTIONS (directional shifts)
────────────────────────────────────────────────────────

"be more firm" → increase decisiveness, remove softeners, make boundary unmistakable
"be more polite" → add warmth, keep core decision, soften delivery
"shorter" → strip to minimum — no added meaning
"ask instead" → convert to question ONLY if natural in context
"be suspicious" → introduce visible doubt or request for explanation
"be direct" → one sentence, no opener, no filler

────────────────────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────────────────────

- ONE reply only. 1 sentence preferred.
- Same language as the original reply
- Sounds like a real local speaker — not a translation
- From the user's perspective (what they say, not what's said to them)
- Vietnamese: User = Anh, Other person = Em or Bạn. Never mix.

FINAL CHECK: "Would a real person type this exact sentence in a chat?"
If not — simplify, shorten, make it natural. Output only the version that passes.

Return ONLY a valid JSON object — no markdown, no extra text:
{ "native": "Reply in original language.", "english": "Translate from the USER's perspective. Boundaries use 'I'. Requests use 'you'. Never render first-person as commands to the user." }`,
        },
        {
          role: "user",
          content: [
            `Original reply:\n${native}`,
            action ? `Locked action: ${action}` : "",
            `User instruction (interpret as intent):\n${instruction}`,
          ].filter(Boolean).join("\n\n"),
        },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, { native, english: native });
    res.json(parsed);

  } catch (err) {
    console.error("/refine error:", err.message);
    res.json({ native: req.body.native, english: req.body.native });
  }
});

// ── /say — natural Vietnamese phrasing ────────────────────────────────────────

app.post("/say", async (req, res) => {
  try {
    const { text, tone } = req.body;

    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: `You help people say things naturally in any language. Write what a real local person would say — idiomatic, correct register.

Detect the language of the user's input and reply in THAT language. Never default to Vietnamese unless the input is in Vietnamese.
Return ONLY a valid JSON object. No markdown.

{
  "native": "Natural phrasing in the user's language, matching the tone.",
  "english": "Translate from the USER's perspective. Boundaries/decisions use 'I'. Requests use 'you'. Never render first-person statements as commands to the user.",
  "toneExplain": "One sentence on how this sounds socially.",
  "variations": {
    "softer": "Softer version.",
    "direct": "More direct version.",
    "shorter": "Shorter version."
  }
}`,
        },
        {
          role: "user",
          content: `What I want to say: "${text}"\nTone: ${tone}`,
        },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, {
      native: "Could not generate a message right now.",
      english: "Could not generate a message right now.",
      toneExplain: "There was an error generating the response.",
      variations: {
        softer: "No variation available.",
        direct: "No variation available.",
        shorter: "No variation available.",
      },
    });

    res.json(parsed);

  } catch (err) {
    console.error("/say error:", err.message);
    res.json({
      native: "An error occurred.",
      english: "An error occurred.",
      toneExplain: err.message,
      variations: {
        softer: "No variation available.",
        direct: "No variation available.",
        shorter: "No variation available.",
      },
    });
  }
});

// ── /continue — continue conversation with context ────────────────────────────

app.post("/continue", async (req, res) => {
  try {
    const { previousMessage, previousReply, previousAnalysis, newMessage } = req.body;

    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: `You are Kova — continuing an ongoing conversation.

You already analyzed a message and suggested a reply. Now the other person has responded.
Do NOT restart from scratch. Continue from where things left off.

CONTEXT RULES:
- Maintain the same tone and strategy unless the new message changes things significantly.
- Only update riskLevel if there is a real reason — not just because there is a new message.
- Keep whatToDo short and actionable.
- The reply (sayThis.native) must be in the same language as the conversation.

Return ONLY a valid JSON object — no markdown, no extra text:
{
  "update": "1–2 lines. What changed or what matters now. Direct, no filler.",
  "riskLevel": "Low" or "Medium" or "High",
  "riskRead": "One sentence under 12 words. Only update if risk actually changed.",
  "whatToDo": ["Short decisive action", "Short decisive action", "Short decisive action"],
  "sayThis": {
    "native": "Reply in the conversation's language. Calm, real, human. Executes whatToDo[0].",
    "english": "Translate from the USER's perspective. Boundaries/decisions use 'I'. Requests use 'you'. Never render first-person statements as commands to the user.",
    "tone": "2–3 words joined with ' • '"
  }
}`,
        },
        {
          role: "user",
          content: `Previous message they sent:
"${previousMessage}"

Your suggested reply:
"${previousReply}"

Previous analysis:
Risk: ${previousAnalysis.riskLevel}
Context: ${previousAnalysis.summary || previousAnalysis.whatThisReallyMeans || ''}

Now they replied:
"${newMessage}"

Continue the conversation.`,
        },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, {
      update: "Could not process continuation.",
      riskLevel: previousAnalysis.riskLevel || "Low",
      riskRead: previousAnalysis.riskRead || "",
      whatToDo: previousAnalysis.whatToDo || [],
      sayThis: { native: "", english: "", tone: "" },
    });

    if (!parsed.sayThis || typeof parsed.sayThis !== "object") {
      parsed.sayThis = { native: "", english: "", tone: "" };
    }
    if (!parsed.sayThis.tone) parsed.sayThis.tone = "";
    if (!Array.isArray(parsed.whatToDo)) parsed.whatToDo = [];
    parsed.whatToDo = dedupeWhatToDo(parsed.whatToDo);

    res.json(parsed);

  } catch (err) {
    console.error("/continue error:", err.message);
    res.json({
      update: "Error processing continuation.",
      riskLevel: "Low",
      riskRead: "",
      whatToDo: [],
      sayThis: { native: "There was an error. Please try again.", english: "There was an error. Please try again.", tone: "" },
    });
  }
});

// ─────────────────────────────────────────────────────────────────────────────

app.listen(3000, "0.0.0.0", () => {
  console.log("Kova backend running on port 3000");
});
