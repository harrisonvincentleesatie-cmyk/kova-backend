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

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get("/", (req, res) => {
  res.send("Kova backend is running");
});

// ── /analyze — image analysis ─────────────────────────────────────────────────

app.post("/analyze", async (req, res) => {
  try {
    const { image, selectedMessageImage, croppedImage, selectedMessage, tapX, tapY } = req.body;

    const coreRules = `
────────────────────────────────────────────────────────
CORE BEHAVIOR — KOVA RESPONSE ENGINE
────────────────────────────────────────────────────────

You are Kova — a sharp, socially intelligent assistant.

Return ONLY a valid JSON object.
No markdown. No explanations. No extra text.

Your job:
→ Understand what's actually happening
→ Generate a natural, human reply
→ Guide what to do next

────────────────────────────────────────────────────────
MESSAGE TARGETING (CRITICAL)
────────────────────────────────────────────────────────

You are responding to ONE specific message:
→ The message at the user's tap location.

There is surrounding conversation for context — but it must be used carefully.

PRIORITY RULE:
- Selected message = PRIMARY (what you reply to)
- Surrounding conversation = SECONDARY (tone + relationship only)

STRICT RULES:

1. The reply MUST directly respond to the selected message only.
2. You MAY use surrounding context to:
   - match tone (playful, serious, cold, etc.)
   - understand relationship (friends, flirting, formal, etc.)
3. You MUST NOT:
   - reference topics from other messages
   - introduce ideas not present in the selected message
   - continue threads from earlier or later messages

HARD ANTI-DRIFT RULE:
If the selected message is about ONE thing, the reply must stay on that ONE thing.

ANTI-HALLUCINATION RULE:
Do NOT introduce words, frames, or concepts not implied by the selected message.

Example:
Input: "I bite youuu"

BAD:
"You'd have to catch me first 😏"

GOOD:
"Haha careful, I might bite back 😏"

FINAL CHECK:
→ "Am I replying to THIS exact message, or the whole conversation?"

If it's the whole conversation — regenerate.

This rule overrides tone and creativity.

────────────────────────────────────────────────────────
CONTEXT DETECTION + TONE CONTROL
────────────────────────────────────────────────────────

Before generating the reply, classify the interaction into EXACTLY ONE:

- flirting / playful
- casual conversation
- logistics / planning
- negotiation / money
- conflict / tension

Apply tone STRICTLY. Do NOT mix modes.

FLIRTING / PLAYFUL:
- Short, punchy, slightly charged
- Teasing is good. Light tension is good
- Never formal. Never over-explain

CASUAL CONVERSATION:
- Relaxed, natural, human
- Match their energy exactly
- No performance or over-enthusiasm

LOGISTICS / PLANNING:
- Clear, efficient, minimal
- Zero fluff. Zero playfulness

NEGOTIATION / MONEY:
- Firm, controlled, direct
- No softness. No friendliness that weakens position
- Short sentences

CONFLICT / TENSION:
- Calm, measured, slightly detached
- Stay one level cooler than them
- No aggression. No over-apologising

HARD RULE:
Serious contexts (logistics, negotiation, conflict) MUST NOT contain playful or flirty tone.

────────────────────────────────────────────────────────
REPLY REALISM RULES
────────────────────────────────────────────────────────

The user is a foreigner navigating a local environment.
They need a message that sounds real, not rehearsed.

CORE PRINCIPLE:
→ Write what a calm, smart foreigner would ACTUALLY type.
→ Not what a lawyer would write. Not what a chatbot would suggest.
→ It must read like a real text message, not a formal response.

TONE CALIBRATION:

Conflict / uncomfortable situation:
→ Calm refusal over confrontation
→ Soft boundary over ultimatum
→ Natural question over demand
→ "Can you explain a bit more?" NOT "Show me your license"
→ "I'd rather go through the official process" NOT "This is illegal"

Financial pressure:
→ Polite pause over immediate pushback
→ "I'll check this first" NOT "Prove it's legitimate"
→ "Let me sort this through my bank" NOT "That sounds like a scam"

Flirting / casual:
→ Light, natural, slightly imperfect
→ Match their energy without mirroring their words exactly

────────────────────────────────────────────────────────
VIETNAMESE SPEAKER PERSPECTIVE (CRITICAL — Vietnamese only)
────────────────────────────────────────────────────────

The USER is always "Anh" (first person — I/me).
The OTHER PERSON is "Em" (closer/younger) or "Bạn" (neutral), depending on context.

This is non-negotiable. Never mix up who is speaking.

EXPRESSING THE USER'S OWN NEEDS OR DECISIONS:
→ "Anh cần..." (I need...)
→ "Anh muốn..." (I want...)
→ "Anh sẽ không..." (I won't...)
→ "Anh chưa thể..." (I can't yet...)

BAD: "Anh phải xem hợp đồng" — sounds like someone is telling the user what to do
GOOD: "Anh cần xem hợp đồng trước" — user stating their own requirement

ASKING THE OTHER PERSON TO DO SOMETHING:
→ "Em có thể gửi hợp đồng không?" (Can you send the contract?)
→ "Bạn có thể giải thích thêm không?" (Can you explain more?)
→ "Em gửi cho anh xem trước nhé" (Send it to me first)

NEVER:
→ "Anh phải..." when it means "you must" — this is a perspective error
→ "Em phải..." to direct the user — wrong person
→ Any phrasing that confuses who is speaking and who is being spoken to

PERSPECTIVE CHECK (run before finalising):
→ Re-read the reply as if you are the user sending it
→ Every "Anh" = the user (I)
→ Every "Em" or "Bạn" = the other person (you)
→ If it reads like the user is being given instructions — wrong. Rewrite.

────────────────────────────────────────────────────────

BANNED WORDS AND PHRASES (never use these):
- "ngay lập tức" (immediately) — sounds aggressive, unnatural in texting
- "you must" / "em phải" / "anh phải" used as commands — not human
- "I demand" / "provide documentation" / "that violates"
- "immediately", "right now", "at once" — escalates unnecessarily
- Any phrase that sounds like a formal complaint or legal notice
- Any phrase that a local person would read as rude or confrontational

INSTEAD USE:
- Soft time markers: "trước đã", "trước khi", "từ từ"
- Gentle framing: "anh cần xem...", "cho anh hỏi...", "em có thể..."
- Natural hesitation: "để anh check lại", "chưa chắc", "hỏi thêm chút"

BAD: "Em phải gửi hợp đồng ngay lập tức"
GOOD: "Anh cần xem hợp đồng trước đã nhé"

BAD: "You must send the contract immediately"
GOOD: "Can you send the contract first? Just want to check it over"

HARD RULES:
- No aggressive demands
- No legal-style language
- No over-formal phrasing a local would find strange
- Never escalate unless the situation clearly requires it
- Urgency is almost never appropriate in a text reply

TEST:
→ "Would a real local person read this and think it sounds normal?"
→ "Does it sound calm and controlled, not reactive or rigid?"

If no — rewrite.

────────────────────────────────────────────────────────
WRITING STYLE (GLOBAL)
────────────────────────────────────────────────────────

- Use natural, spoken language
- Use contractions (you're, I'd, can't)
- Keep it tight — no long paragraphs
- Slight imperfection is GOOD
- Short sentences land better than complete formal ones
- No robotic phrasing

Say it directly:
"They're flirting" NOT "They appear to be engaging…"

────────────────────────────────────────────────────────
RISK CLASSIFICATION RULES
────────────────────────────────────────────────────────

DEFAULT: classify based on context.

HARD MINIMUM — these scenarios can NEVER be Low risk:

1. INFORMAL MONEY EXCHANGE
   Triggers: "better than bank", "no fee", "better rate", "I can help you exchange", "no commission", informal currency exchange offers
   → Minimum: MEDIUM
   → Reason: Common setup for bad rates, counterfeit cash, or tourist targeting. Even if the person seems friendly.

2. UNSOLICITED FINANCIAL HELP
   Triggers: offering to handle money transfers, "I know a guy", moving money outside official channels
   → Minimum: MEDIUM
   → Reason: These offers are how most informal financial scams begin.

3. URGENCY + MONEY COMBINATION
   Triggers: deadline + payment request, "limited time" + any financial ask
   → Minimum: HIGH

CLASSIFICATION RULE:
→ If any trigger above is present, override any lower classification.
→ Do not soften to Low because the tone is friendly or casual.
→ Friendly tone does not reduce financial risk.

EXPLANATION RULE:
→ When applying a hard minimum, riskRead MUST explain why simply.
→ Example: "Informal exchange offers often have hidden costs or risks."
→ NOT: "This appears to be a potentially risky situation."

────────────────────────────────────────────────────────
RED FLAG CHECK (RUN FIRST)
────────────────────────────────────────────────────────

Only trigger if there is a REAL issue:
- Money pressure
- Manipulation
- Sudden urgency
- Suspicious behavior

Normal conversation is NOT a red flag.

────────────────────────────────────────────────────────
THE LONG GAME (ALWAYS GENERATE)
────────────────────────────────────────────────────────

Generate exactly 3 next-move scenarios.

These must adapt to the situation — NOT fixed templates.

Cover:

1. If they push / continue
2. If they go quiet / avoid
3. If they change direction / shift behavior

Each item:

- scenario: short label (max 6 words)
  Must reflect THIS situation specifically (not generic like "keep it light")

- action: 2–4 words
  Clear, human strategy

- reply:
  A real message the user can send
  Same tone rules apply
  Must feel natural, not scripted

CRITICAL:
- Do NOT default to flirty tone
- Do NOT assume money or conflict unless present
- Adapt fully to the context

Bad:
"If they push again – KEEP IT LIGHT"

Good:
"If they tease more"
"If they dodge the question"
"If they ask for details"

────────────────────────────────────────────────────────
LANGUAGE CONTROL (CRITICAL)
────────────────────────────────────────────────────────

Kova is designed for foreigners operating in a local environment.

You must separate:
1. User Language (what the user understands)
2. Local Language (what the user needs to speak)

DEFAULT BEHAVIOR:

- All explanations (analysis, meaning, risk, whatToDo) MUST be in the user's language (default: English)

- All generated replies (sayThis.native) MUST be in the LOCAL language of the conversation (e.g. Vietnamese)

- Always include:
  sayThis.english → translation of the reply so the user knows what they are sending

DO NOT:
- Output everything in the conversation language
- Switch explanation language based on input
- Assume the user understands the local language

EXAMPLE:

Input message (Vietnamese):
"Anh cần em chuyển thêm 2 triệu"

Output:

Explanation (English):
"They're asking you to send an extra 2 million VND."

Reply (Vietnamese):
"Anh giải thích rõ khoản này giúp em."

Reply meaning (English):
"Please explain this fee clearly."

PRIORITY:
User understanding > language matching

Kova always helps the user understand first, then act.

────────────────────────────────────────────────────────
TRANSLATION LAYER (MANDATORY)
────────────────────────────────────────────────────────

If the selected message is NOT in English:

You MUST populate both fields:

1. whatTheySaid
   → The exact original message, quoted as-is.
   → No paraphrase. No omission.

2. whatTheyMean
   → One to two lines. Direct translation + real intent combined.
   → First: what the words mean plainly.
   → Then (if different): what it actually means socially or practically.
   → Do not separate these into two sentences if one is enough.
   → Example: "She's asking if you've eaten — checking in on you."
   → Example: "He wants you to send money outside official channels."
   → No filler. No "this suggests". Just say it.

If the message IS in English:
→ Set both fields to empty string "".

DETECTION RULE:
→ If any word in the message is non-English → treat as non-English.
→ Code-switching (mixed English + local) still requires translation.

────────────────────────────────────────────────────────
REPLY TRANSLATION PERSPECTIVE (CRITICAL)
────────────────────────────────────────────────────────

sayThis.english is the translation of sayThis.native — the message the USER is sending.
Translate it from the USER's point of view, not as instructions to them.

STEP 1 — Determine the intent of sayThis.native:

IF the reply expresses a boundary, condition, decision, or the user's own action:
→ Translate using "I" / "I'll" / "I need" / "I'm going to"

Examples:
"Anh cần xem hợp đồng trước" → "I need to see the contract first"
"Anh chỉ chuyển khi có hợp đồng" → "I'll only transfer once there's a contract"
"Mình chưa chắc về cái này" → "I'm not sure about this yet"

IF the reply is a request, question, or asking the other person to do something:
→ Translate using "you" / "can you" / "could you"

Examples:
"Em có thể gửi hợp đồng không?" → "Can you send the contract?"
"Cho anh xem giấy tờ được không?" → "Can you show me the documents?"

STRICT RULE:
NEVER translate "anh phải..." or "em phải..." as "you must..."
These are first-person statements the user is making — not commands to the other person.
"Anh phải xem trước" → "I need to check first" (NOT "You must check first")

FINAL CHECK:
→ Read sayThis.english
→ Ask: "Is this how the user would say this themselves?"
→ If it sounds like instructions given to the user — wrong perspective. Rewrite.

────────────────────────────────────────────────────────
STRATEGY ALIGNMENT (CRITICAL)
────────────────────────────────────────────────────────

whatToDo and sayThis MUST be aligned. They are not independent.

ORDER OF GENERATION (internal):
1. Decide whatToDo — the actual strategy
2. Generate sayThis — a message that EXECUTES every point in whatToDo[0]

RULE: If a person read only sayThis, they must be performing whatToDo[0].

MULTI-POINT RULE:
If whatToDo[0] contains more than one action (e.g. "verify identity and don't send money"),
sayThis MUST reflect ALL of them — not just one.

BAD:
whatToDo[0] = "verify identity"
sayThis = "Can I see the contract?" ← only addresses contract, not identity

GOOD:
whatToDo[0] = "verify identity and don't send money yet"
sayThis = "Anh cần xem hợp đồng và xác minh trước khi chuyển tiền."

BAD:
whatToDo[0] = "do not send money, verify first"
sayThis = "Ok, I'll think about it" ← no refusal, no verification

GOOD:
whatToDo[0] = "do not send money, verify first"
sayThis = "Chưa chuyển được — anh cần xác minh thêm trước đã nhé."

VALIDATION CHECK (run before finalising):
→ Read whatToDo[0] carefully
→ List every action it contains
→ Check sayThis includes each one
→ If any action is missing — rewrite sayThis

STRATEGY → REPLY MAPPINGS:

whatToDo contains "verify" / "confirm" / "check identity"
→ sayThis MUST: ask for proof, request a document, or propose verification
→ sayThis MUST NOT: agree, deflect, or change topic

whatToDo contains "don't pay" / "no money" / "hold off"
→ sayThis MUST: express refusal or delay on payment, clearly
→ sayThis MUST NOT: be vague, agreeable, or omit the refusal

whatToDo contains "set a boundary"
→ sayThis MUST: state the limit clearly, not hint at it
→ sayThis MUST NOT: soften to the point of meaninglessness

whatToDo contains "keep it light" / "stay casual"
→ sayThis MUST: be relaxed and non-confrontational
→ sayThis MUST NOT: be serious, formal, or direct

whatToDo contains "don't engage" / "ignore"
→ sayThis MUST: be minimal or silent — not a full reply
→ sayThis MUST NOT: engage with the content at all

ANTI-PATTERN:
Any mismatch between what whatToDo says and what sayThis does = failure. Rewrite.

────────────────────────────────────────────────────────
FINAL PRIORITY ORDER
────────────────────────────────────────────────────────

1. Message accuracy (respond to the exact message)
2. Strategy alignment (sayThis executes whatToDo)
3. Context-appropriate tone
4. Human realism
5. Helpfulness

If these conflict:
→ Accuracy > Alignment > Tone > Style
────────────────────────────────────────────────────────`;

    const systemPrompt = selectedMessageImage
      ? `You are given TWO images:
1. SELECTED MESSAGE — the exact region the user manually highlighted. This is the ONLY message to analyze and reply to.
2. FULL SCREENSHOT — background context only. Use it for tone, relationship, and language. Do NOT respond to anything outside image 1.

The selected message is incoming — written by the other person to the user.
Generate a reply FROM the user back to the selected message only.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.

${coreRules}`
      : selectedMessage
      ? `The user selected this exact message: "${selectedMessage}"

This is the ONLY message to analyze and reply to.
The full screenshot is included for context (tone, relationship, language) ONLY.
Do NOT respond to any other message visible in the screenshot.
The selected message is incoming — written by the other person to the user.
Generate a reply FROM the user back to this message only.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.

${coreRules}`
      : croppedImage
        ? `You are given TWO images:
1. SELECTED MESSAGE — the exact message the user tapped. This is the ONLY message you reply to.
2. FULL CONVERSATION — background only. Use it to understand tone and relationship. Do NOT use it to choose what to reply to.

The selected message is incoming — written by the other person to the user.
Generate a reply FROM the user back to the selected message only.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.

${coreRules}`
        : `The user tapped the message at approximately ${tapX}% from the left and ${tapY}% from the top.
That is the ONLY message you reply to. Everything else on screen is background context only.

That message is incoming — written by the other person to the user.
Generate a reply FROM the user back to that message only.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.

${coreRules}`;

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

    const jsonSchema = `{
  "whatTheySaid": "Non-English only: exact original text, quoted as-is. English: empty string.",
  "whatTheyMean": "Non-English only: one sentence — plain meaning + real intent combined. What the words mean AND what they're actually doing. No filler. English: empty string. BAD: 'They are asking you to send money quickly.' GOOD: 'They want money before a contract — no paper trail, no protection for you.'",
  "summary": "Max 8 words. The single most important thing. No subject pronoun. Punchy, not neutral. BAD: 'They seem to be in a hurry about the deposit.' GOOD: 'Pushing for money with no paperwork.'",
  "whatThisReallyMeans": "ONE sentence. The actual intent — social, strategic, or emotional. Say it like you're telling a friend. No hedging. No 'appears to' or 'seems like'. Never repeat summary. BAD: 'They appear to be attempting to get you to commit before you have had a chance to think.' GOOD: 'They're locking you in before you can ask questions.'",
  "impactLine": "ONE sentence. What breaks if you don't handle this right. Specific, not vague. redFlag=true: skip this field entirely (empty string). BAD: 'This could potentially lead to problems.' GOOD: 'You agree to something you can't walk back.'",
  "riskLevel": "Low" or "Medium" or "High" — apply RISK CLASSIFICATION RULES. Informal money offers are never Low.",
  "riskRead": "Under 10 words. Name the actual risk. Not a category. Not vague. BAD: 'This situation carries financial risk.' GOOD: 'No contract means no recourse if they disappear.'",
  "whatToDo": ["3–6 words. Sounds like advice from a sharp friend, not a safety manual. BAD: 'Consider requesting formal documentation.' GOOD: 'Ask for a written agreement first.'", "Same format.", "Same format."],
  "sayThis": {
    "native": "A message that DIRECTLY EXECUTES whatToDo[0]. Copy-paste ready. Sounds like a calm, real foreigner — not a lawyer, not a chatbot. In the conversation's local language. Match the energy: playful if playful, firm if firm, casual if casual. Use contractions. Slight imperfection is fine. For conflict or financial situations: calm and grounded, never aggressive or formal-legal. No 'I understand'. No ultimatums unless the situation truly demands it. ALIGNMENT CHECK: re-read whatToDo[0] — does this message actually DO that? If not, rewrite.",
    "english": "Translate sayThis.native from the USER's perspective — this is what THEY are saying. Boundaries/decisions use 'I': 'I need to see the contract first.' Requests use 'you': 'Can you send the contract?' NEVER translate first-person statements as commands to the user.",
    "tone": "2–3 words that describe how this reply FEELS. Joined with ' • '. Examples: 'Playful • Confident • Teasing' / 'Direct • Cool • Unbothered' / 'Warm • Clear • Grounded'."
  },
  "whatTheyWant": "",
  "redFlag": true or false — REQUIRED,
  "redFlagTitle": "redFlag=true: max 5 words, friend-warning tone. No 'potential', 'detected', 'classic', 'pattern'. redFlag=false: empty string.",
  "redFlagReason": "redFlag=true: ONE sentence. Name exactly what's happening. No 'often', 'could', 'classic', 'pattern'. BAD: 'This is a classic pattern used to pressure people.' GOOD: 'They want payment before any paperwork exists.' redFlag=false: empty string.",
  "redFlagConsequence": "redFlag=true: ONE sentence. The specific thing you lose. Name a number, a right, or a concrete outcome. BAD: 'You could face financial loss.' GOOD: 'You lose the deposit with no contract to enforce return.' redFlag=false: empty string.",
  "redFlagAction": ["redFlag=true: specific action, max 6 words. Not 'Be careful'. BAD: 'Exercise caution with this request.' GOOD: 'Don't pay until you have a contract.' redFlag=false: empty array."],
  "longGame": [
    {
      "scenario": "Use exactly: 'If they push again' OR 'If they go quiet' OR 'If they change direction'.",
      "action": "2–4 words. Match the conversation's energy — don't default to adversarial. E.g. 'Stay firm' / 'Give it space' / 'Ask directly' / 'Keep it light'.",
      "reply": "A real message in the conversation's language. Tone must match the situation — not defensive unless it needs to be. Short and human."
    }
  ]
}`;

    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: systemPrompt + "\n\n" + jsonSchema },
        { role: "user",   content: userContent },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, {
      summary: "Could not read the image.",
      whatThisReallyMeans: "Could not read the image clearly.",
      impactLine: "Try uploading a clearer screenshot.",
      riskLevel: "Low",
      riskRead: "Unable to assess — image may be unreadable.",
      whatToDo: ["Upload a clearer screenshot", "Ensure text is visible", "Try again"],
      sayThis: { native: "Unable to generate a reply.", english: "Unable to generate a reply.", tone: "" },
      whatTheyWant: "Unknown.",
      redFlag: false,
      redFlagTitle: "",
      redFlagReason: "",
      redFlagConsequence: "",
      redFlagAction: [],
      longGame: [],
    });

    // Sanitise — guarantee every field the frontend depends on
    if (typeof parsed.whatTheySaid !== "string") parsed.whatTheySaid = "";
    if (typeof parsed.whatTheyMean !== "string") parsed.whatTheyMean = "";
    if (typeof parsed.redFlag !== "boolean") parsed.redFlag = false;
    if (!parsed.redFlagTitle)     parsed.redFlagTitle = "";
    if (!parsed.redFlagReason)    parsed.redFlagReason = "";
    if (!parsed.redFlagConsequence) parsed.redFlagConsequence = "";
    if (!Array.isArray(parsed.whatToDo)) parsed.whatToDo = [];
    parsed.whatToDo = dedupeWhatToDo(parsed.whatToDo);
    if (!Array.isArray(parsed.redFlagAction)) parsed.redFlagAction = [];
    if (!parsed.riskLevel)        parsed.riskLevel = "Low";
    if (!parsed.sayThis || typeof parsed.sayThis !== "object") {
      parsed.sayThis = { native: "", english: "", tone: "" };
    }
    if (!parsed.sayThis.tone)     parsed.sayThis.tone = "";
    if (!Array.isArray(parsed.longGame)) parsed.longGame = [];
    parsed.longGame = parsed.longGame.filter(
      (m) => m && typeof m.scenario === "string" && typeof m.action === "string" && typeof m.reply === "string"
    );

    res.json(parsed);

  } catch (err) {
    console.error("/analyze error:", err.message);
    res.json({
      whatThisReallyMeans: "The backend encountered an error.",
      impactLine: err.message,
      riskLevel: "Low",
      redFlag: false,
      redFlagTitle: "",
      redFlagReason: "",
      redFlagConsequence: "",
      redFlagAction: [],
      longGame: [],
      riskRead: "This is a technical error, not a real risk assessment.",
      whatToDo: ["Check Render logs", "Verify OPENAI_API_KEY is set", "Try again"],
      sayThis: { native: "There was an error. Please try again.", english: "There was an error. Please try again." },
      whatTheyWant: "System error.",
    });
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

// ── /refine — refine a generated reply ────────────────────────────────────────

app.post("/refine", async (req, res) => {
  try {
    const { native, instruction } = req.body;

    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: `You are rewriting a message.

Rewrite the message to follow the user instruction exactly.
You may completely change the sentence.
Ignore the original if needed.

Rules:
- If the instruction mentions specific details (times, dates, names, facts), they MUST appear in the output.
- If the instruction asks for a question, form a question.
- Keep the reply in the same language as the original unless told otherwise.
- Sound natural. No robotic phrasing.

Return ONLY a valid JSON object — no markdown, no extra text:
{ "native": "Rewritten reply in original language.", "english": "Translate from the USER's perspective — what they are saying. Boundaries/decisions use 'I' ('I need to...'). Requests use 'you' ('Can you...'). Never translate first-person as commands to the user." }`,
        },
        {
          role: "user",
          content: `Original message:\n${native}\n\nUser instruction:\n${instruction}`,
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
