import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

const openai = new OpenAI({
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
    const { image, croppedImage, tapX, tapY } = req.body;

    const writingRules = `
RULES FOR EVERY FIELD:
- Speak directly to the user. Say "they" for the other person, "you" for the user. Never say "the sender" or "the user".
- No hedging. Never write "could", "may", "might", "potentially", "it seems". Say what it is.
- No filler. Never start with "This indicates", "It appears", "Based on".
- Be concise. Cut any word that doesn't add meaning.
- Sound human. Write like a sharp friend, not a customer service bot.`;

    const redFlagRules = `
─── RED FLAG CHECK (run before everything else) ─────────────────────────────

Always return: redFlag, redFlagTitle, redFlagReason, redFlagAction.

TRIGGERS — flag immediately if ANY of these appear:
• Price or amount changed without explanation
• Urgency: "pay now", "last chance", "today only", "by end of day"
• A fee that wasn't mentioned before
• Request to pay someone other than the expected party
• Emotional pressure used to rush a decision (guilt, flattery, fear)
• Dodging a direct question about price, timeline, or terms
• Info that contradicts something said earlier

WHEN redFlag = true:
- redFlagTitle: max 5 words. Sounds like a smart friend talking. Examples: "Fee wasn't mentioned before" / "They're rushing you" / "Price changed with no reason" / "This doesn't add up". No percentages. No "potential". No "detected".
- redFlagReason: ONE sentence only. Name the tactic directly. Include local context if useful — e.g. "Maintenance fees go to building management, not the landlord." No hedging. No "this could be", no "often seen in scams".
- redFlagAction: 1–2 specific things the user can do right now. Concrete. Examples: ["Ask for a written breakdown before paying", "Don't transfer until you have a receipt"]. Not: ["Be careful"].

WHEN redFlag = true AND whatThisReallyMeans would repeat the same point:
- Keep whatThisReallyMeans to ONE short sentence covering what the red flag doesn't already say.

WHEN redFlag = false:
- redFlagTitle = "", redFlagReason = "", redFlagAction = []

Normal negotiation, small delays, and polite pressure are NOT red flags.`;

    const systemPrompt = croppedImage
      ? `You are Kova — a sharp social intelligence engine. Return ONLY a valid JSON object — no markdown, no extra text.

You are given TWO images:
1. SELECTED MESSAGE — the cropped region the user tapped. This is the ONLY message you analyse and reply to.
2. FULL CONVERSATION — for tone, relationship, and context only.

The selected message is incoming — written by the other person to the user.
Generate a reply FROM the user back to that message.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.
${writingRules}
${redFlagRules}`
      : `You are Kova — a sharp social intelligence engine. Return ONLY a valid JSON object — no markdown, no extra text.

Focus on the message at approximately ${tapX}% from the left and ${tapY}% from the top.
That message is incoming — written by the other person to the user.
Generate a reply FROM the user back to that message.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.
${writingRules}
${redFlagRules}`;

    const userContent = [
      { type: "text", text: croppedImage ? "Selected message (image 1). Full conversation (image 2)." : "Analyse this screenshot." },
    ];
    if (croppedImage) {
      userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${croppedImage}` } });
    }
    userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${image}` } });

    const jsonSchema = `{
  "summary": "One line, max 10 words. The single most important thing about this message. No subject pronoun needed. Examples: 'Routine charge — nothing unusual' / 'Delaying — no clear timeline given' / 'Polite but non-committal'.",
  "whatThisReallyMeans": "What they're actually doing socially — not a literal description. If redFlag=true, keep this to ONE sentence covering what the red flag didn't already explain. If redFlag=false, 1–2 sharp sentences. Start with what's really happening, not what was said.",
  "impactLine": "What goes wrong if you respond badly. One direct sentence. No hedging.",
  "riskLevel": "Low" or "Medium" or "High",
  "riskRead": "One decisive sentence, under 12 words. Say what the risk actually is.",
  "whatToDo": ["Short confident action", "Short confident action", "Short confident action"],
  "sayThis": {
    "native": "A natural reply the user can send. In the conversation's language. Sound like a real person — no over-politeness, no robot phrasing. Match the tone of the conversation.",
    "english": "Plain English meaning. If already English, rephrase slightly so it doesn't feel like a repeat.",
    "tone": "2–3 short human descriptors for the tone of the reply, joined with ' • '. Examples: 'Direct • Calm • Controlled' / 'Polite • Warm • Clear' / 'Playful • Light • Easy'. No technical jargon."
  },
  "whatTheyWant": "",
  "redFlag": true or false — REQUIRED. Always present.
  "redFlagTitle": "If redFlag=true: short punchy human title, max 5 words. If false: empty string.",
  "redFlagReason": "If redFlag=true: one sharp sentence with local context. If false: empty string.",
  "redFlagAction": ["If redFlag=true: 1–2 specific actions. If false: empty array."]
}`;

    const response = await openai.chat.completions.create({
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
      redFlagAction: [],
    });

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
      redFlagAction: [],
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

    const response = await openai.chat.completions.create({
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

    const response = await openai.chat.completions.create({
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
