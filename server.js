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

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get("/", (req, res) => {
  res.send("Kova backend is running");
});

// ── /analyze — image analysis ─────────────────────────────────────────────────

app.post("/analyze", async (req, res) => {
  try {
    const { image, croppedImage, selectedMessage, tapX, tapY } = req.body;

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
HUMAN EDGE / IMPACT RULES
────────────────────────────────────────────────────────

This rule is context-sensitive:
- STRONG in flirting / casual conversation
- MODERATE in normal conversation
- MINIMAL in logistics / negotiation / conflict

Never sacrifice clarity or trust for "edge" in serious contexts.

The reply must feel like a real person — not an assistant.

Avoid:
- Generic responses anyone could say
- Passive or overly agreeable replies
- Filler like "haha that's funny", "I'll be careful"

Prefer:
- Slight tension
- Playful challenge (if appropriate)
- Confidence over politeness
- Personality over perfection

GUIDELINES:

1. Don't just react — add something
2. Avoid predictable phrasing
3. Slight boldness is GOOD when context allows
4. Replies should feel specific, not templated

Test:
→ "Would a real, confident person actually send this?"

If not — regenerate.

────────────────────────────────────────────────────────
WRITING STYLE (GLOBAL)
────────────────────────────────────────────────────────

- Use natural, spoken language
- Use contractions (you're, I'd, can't)
- Keep it tight — no long paragraphs
- Slight imperfection is GOOD
- No robotic phrasing

Say it directly:
"They're flirting" NOT "They appear to be engaging…"

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
FINAL PRIORITY ORDER
────────────────────────────────────────────────────────

1. Message accuracy (respond to the exact message)
2. Context-appropriate tone
3. Human realism
4. Helpfulness

If these conflict:
→ Accuracy > Tone > Style
────────────────────────────────────────────────────────`;

    const systemPrompt = selectedMessage
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

    const userContent = [
      { type: "text", text: croppedImage ? "Selected message (image 1). Full conversation (image 2)." : "Analyse this screenshot." },
    ];
    if (croppedImage) {
      userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${croppedImage}` } });
    }
    userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${image}` } });

    const jsonSchema = `{
  "summary": "One line, max 10 words. Most important thing about this message. No subject pronoun. Sharp and direct — not neutral. Examples: 'Rushing you before you can think' / 'Keeping it light, testing the waters' / 'Dodging the actual question'.",
  "whatThisReallyMeans": "What's actually happening socially — said directly, like a friend would say it out loud. If redFlag=true: ONE sentence about something the red flag didn't already cover. If redFlag=false: 1–2 sentences max. No 'they are likely', no 'it appears', no 'they seem to be'. Say it: 'They're flirting.' 'They're stalling.' 'They want you to act before you think.'",
  "impactLine": "If redFlag=true: 5 words max or skip entirely. If redFlag=false: what goes wrong if you don't handle this right. One short sentence. Direct.",
  "riskLevel": "Low" or "Medium" or "High",
  "riskRead": "One sentence, under 12 words. Say the actual risk — not a category.",
  "whatToDo": ["Short decisive action — sounds like what you'd tell a friend", "Same", "Same"],
  "sayThis": {
    "native": "A reply the user would ACTUALLY send. In the conversation's language. Match the energy exactly — playful if playful, firm if firm, casual if casual. Use contractions. Slight imperfection is fine. No over-politeness. No 'I understand'. No full formal sentences unless the chat is clearly formal.",
    "english": "Plain English meaning. If already English, rephrase slightly — don't just repeat.",
    "tone": "2–3 words that describe how this reply FEELS. Joined with ' • '. Examples: 'Playful • Confident • Teasing' / 'Direct • Cool • Unbothered' / 'Warm • Clear • Grounded'."
  },
  "whatTheyWant": "",
  "redFlag": true or false — REQUIRED,
  "redFlagTitle": "redFlag=true: max 5 words, friend-warning tone. No 'potential', 'detected', 'classic', 'pattern'. redFlag=false: empty string.",
  "redFlagReason": "redFlag=true: ONE line. State what it is — not what it 'often means'. No 'classic', 'common', 'pattern', 'often', 'could'. redFlag=false: empty string.",
  "redFlagConsequence": "redFlag=true: REQUIRED — ONE line, always. The specific stake. Name an amount, contract, or concrete loss. No vague language. redFlag=false: empty string.",
  "redFlagAction": ["redFlag=true: 1–2 immediate specific actions. Not 'Be careful'. redFlag=false: empty array."],
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
    if (typeof parsed.redFlag !== "boolean") parsed.redFlag = false;
    if (!parsed.redFlagTitle)     parsed.redFlagTitle = "";
    if (!parsed.redFlagReason)    parsed.redFlagReason = "";
    if (!parsed.redFlagConsequence) parsed.redFlagConsequence = "";
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

// ── /refine — refine a generated reply ────────────────────────────────────────

app.post("/refine", async (req, res) => {
  try {
    const { native, english, instruction } = req.body;

    const response = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: `You refine conversation replies. Apply the instruction and return ONLY a valid JSON object — no markdown, no extra text.

Keep the reply in the same language as the original "native" text unless told otherwise.
Sound like a real person. No over-politeness, no robotic phrasing. Match the tone of the original.

{ "native": "Refined reply in original language.", "english": "Plain English meaning. Rephrase if already English." }`,
        },
        {
          role: "user",
          content: `Current reply: "${native}"\nEnglish meaning: "${english}"\nInstruction: ${instruction}`,
        },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, { native, english });
    res.json(parsed);

  } catch (err) {
    console.error("/refine error:", err.message);
    res.json({ native: req.body.native, english: req.body.english });
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
  "english": "Plain English meaning. Rephrase if already English.",
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

// ─────────────────────────────────────────────────────────────────────────────

app.listen(3000, "0.0.0.0", () => {
  console.log("Kova backend running on port 3000");
});
