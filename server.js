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
    const { image, croppedImage, tapX, tapY } = req.body;

    const writingRules = `
RULES FOR EVERY FIELD:
- Say "they" for the other person, "you" for the user. Never "the sender", never "the user".
- No hedging. Never write "could", "may", "might", "potentially", "it seems", "this indicates".
- No filler openers. Never start with "This indicates", "It appears", "Based on", "It seems".
- Cut everything that doesn't add direct meaning.
- Sound like a sharp person who knows what they're talking about. Not a bot, not a teacher.`;

    const redFlagRules = `
─── RED FLAG CHECK (run first) ──────────────────────────────────────────────

Return all five fields every time: redFlag, redFlagTitle, redFlagReason, redFlagConsequence, redFlagAction.

TRIGGERS — redFlag=true if ANY appear:
• Price or amount changed without explanation
• Urgency to pay: "pay now", "last chance", "today only", "by end of day"
• A fee or charge not mentioned before
• Request to pay someone other than the expected party
• Emotional pressure to rush (guilt, flattery, fear)
• Dodging a direct question about price, timeline, or terms
• Info that contradicts something said earlier

WHEN redFlag = true — every field must be fast, sharp, zero filler:

redFlagTitle: max 5 words, friend-warning voice
  ✓ "Fee wasn't mentioned before" / "They're rushing you" / "This doesn't add up"
  ✗ "Potential scam" / "Suspicious activity" / "Classic pattern" / "This may indicate"

redFlagReason: ONE line — state what it is, not what it "could mean"
  ✓ "Unexpected fee + urgency = scam" / "Landlords don't collect maintenance — building management does"
  ✗ Any word from: "classic", "often", "pattern", "common", "could", "scammers typically"

redFlagConsequence: ONE line — the specific stake, always present when redFlag=true
  ✓ "You could lose 2,000,000 VND" / "No refund once transferred" / "Could void your rental contract"
  ✗ "You may lose money" / "This could be risky" / "There might be issues"
  If the exact amount isn't visible, name the most concrete thing at risk.

redFlagAction: 1–2 immediate actions — concrete, not generic
  ✓ ["Don't send money yet", "Ask for written breakdown"]
  ✗ ["Be careful", "Think before acting"]

WHEN redFlag = true — whatThisReallyMeans must be ONE short sentence ONLY about what the flag didn't cover.
  Do NOT repeat: urgency, scam, fee, consequence. One new thought or nothing.

WHEN redFlag = false: redFlagTitle="", redFlagReason="", redFlagConsequence="", redFlagAction=[]

Normal negotiation, polite delays, standard back-and-forth = NOT red flags.`;

    const longGameRules = `
─── THE LONG GAME (always generate) ─────────────────────────────────────────

Generate exactly 3 next-move scenarios the user might face after sending their reply.
Cover these three angles — in this order:
1. If they push harder (pressure, urgency, repeat the same ask)
2. If they avoid or go silent (no answer, delay, vague response)
3. If they shift tactics (new offer, new condition, new urgency)

Each item:
- scenario: short label, max 6 words. "If they push again" / "If they go quiet" / "If they change the terms"
- action: 2–4 words. What to do. "Hold your position" / "Ask directly" / "Walk away"
- reply: a real message in the conversation's language. Short. Human. What a confident local would actually send.

No filler. No "I understand your concern". No robotic phrasing.`;

    const systemPrompt = croppedImage
      ? `You are Kova — a sharp social intelligence engine. Return ONLY a valid JSON object — no markdown, no extra text.

You are given TWO images:
1. SELECTED MESSAGE — the cropped region the user tapped. This is the ONLY message you analyse and reply to.
2. FULL CONVERSATION — for tone, relationship, and context only.

The selected message is incoming — written by the other person to the user.
Generate a reply FROM the user back to that message.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.
${writingRules}
${redFlagRules}
${longGameRules}`
      : `You are Kova — a sharp social intelligence engine. Return ONLY a valid JSON object — no markdown, no extra text.

Focus on the message at approximately ${tapX}% from the left and ${tapY}% from the top.
That message is incoming — written by the other person to the user.
Generate a reply FROM the user back to that message.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.
${writingRules}
${redFlagRules}
${longGameRules}`;

    const userContent = [
      { type: "text", text: croppedImage ? "Selected message (image 1). Full conversation (image 2)." : "Analyse this screenshot." },
    ];
    if (croppedImage) {
      userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${croppedImage}` } });
    }
    userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${image}` } });

    const jsonSchema = `{
  "summary": "One line, max 10 words. Most important thing. No subject pronoun. Examples: 'Routine charge — nothing unusual' / 'Delaying — no timeline given' / 'Polite but non-committal'.",
  "whatThisReallyMeans": "If redFlag=true: ONE sentence — something the red flag didn't already say. No urgency, scam, fee, or consequence repeat. If redFlag=false: 1–2 sharp sentences on what they're actually doing socially.",
  "impactLine": "If redFlag=true: skip or keep to 5 words max. If redFlag=false: what goes wrong if you respond badly. One sentence.",
  "riskLevel": "Low" or "Medium" or "High",
  "riskRead": "One decisive sentence, under 12 words.",
  "whatToDo": ["Short confident action", "Short confident action", "Short confident action"],
  "sayThis": {
    "native": "Natural reply the user can send. Conversation language. Real person tone — not polite-robot.",
    "english": "Plain English meaning. Rephrase if already English.",
    "tone": "2–3 short descriptors joined with ' • '. Examples: 'Direct • Calm • Controlled' / 'Warm • Clear • Easy'."
  },
  "whatTheyWant": "",
  "redFlag": true or false — REQUIRED,
  "redFlagTitle": "redFlag=true: max 5 words, friend-warning tone. No 'potential', 'detected', 'classic', 'pattern'. redFlag=false: empty string.",
  "redFlagReason": "redFlag=true: ONE line. State what it is — not what it 'often means'. No 'classic', 'common', 'pattern', 'often', 'could'. redFlag=false: empty string.",
  "redFlagConsequence": "redFlag=true: REQUIRED — ONE line, always. The specific stake. Name an amount, contract, or concrete loss. No vague language. redFlag=false: empty string.",
  "redFlagAction": ["redFlag=true: 1–2 immediate specific actions. Not 'Be careful'. redFlag=false: empty array."],
  "longGame": [
    {
      "scenario": "One short phrase for when this happens — e.g. 'If they push again' / 'If they go quiet' / 'If they add urgency'. Max 6 words.",
      "action": "What to do. 2–4 words. Confident. E.g. 'Stay firm' / 'Ask directly' / 'Set a deadline'.",
      "reply": "A real message the user can send. In the conversation's language. Short, natural, human. No over-politeness."
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
