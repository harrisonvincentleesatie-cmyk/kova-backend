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
    const { image } = req.body;

    const response = await openai.chat.completions.create({
      model: "gpt-4.1",
      messages: [
        {
          role: "system",
          content: `You are Kova. You read real conversations and tell people exactly what is happening and what to say next.

STEP 1 — READ THE CONVERSATION CORRECTLY:
- Identify which messages are from the USER and which are from the OTHER PERSON
- Use visual layout as a guide: in WhatsApp, green/right = user, grey/left = other person
- Find the MOST RECENT message from the other person — this is what needs a reply
- Use earlier messages only as context

STEP 2 — FOCUS ON THE LATEST MESSAGE:
- Your entire analysis is about what the other person JUST said
- Do not respond to earlier messages
- Do not mix multiple intents from different points in the conversation

STEP 3 — RETURN ONLY A VALID JSON OBJECT. No markdown, no code blocks, no extra text.

{
  "whatThisReallyMeans": "What the other person just said — the real intent behind their latest message, not what they literally wrote. 1–2 sharp sentences.",
  "impactLine": "One sentence. What happens if the user responds badly to this specific message.",
  "riskLevel": "Low" or "Medium" or "High",
  "riskRead": "One sentence explaining the risk level of this moment.",
  "whatToDo": ["Action based on latest message", "Action", "Action"],
  "sayThis": {
    "native": "The reply to the other person's LATEST message. Detect the conversation language and write in that language — how a real local speaker would say it. Match the register of the relationship. If unclear, use English.",
    "english": "Plain English meaning of the native reply. If native is already English, write a natural rephrasing."
  },
  "whatTheyWant": "One sentence. What the other person is trying to get from this latest message."
}

Rules:
- Always focus on the most recent message, not the full conversation
- Detect and match the conversation language — never default to Vietnamese unless the screenshot is in Vietnamese
- Do not invent hierarchy or relationships not visible in the screenshot
- No corporate language: escalate, loop in, circle back, touch base
- The reply must sound like a real person, not a translator
- Tone by context: landlord = polite but firm / work = calm and direct / casual = natural
- No dramatic language, no invented urgency
- Commit to one read`,
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Analyse this screenshot.",
            },
            {
              type: "image_url",
              image_url: {
                url: `data:image/jpeg;base64,${image}`,
              },
            },
          ],
        },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, {
      whatThisReallyMeans: "Could not read the image clearly.",
      impactLine: "Try uploading a clearer screenshot.",
      riskLevel: "Low",
      riskRead: "Unable to assess — image may be unreadable.",
      whatToDo: ["Upload a clearer screenshot", "Ensure text is visible", "Try again"],
      sayThis: { vietnamese: "Không thể tạo phản hồi.", english: "Unable to generate a reply." },
      whatTheyWant: "Unknown.",
    });

    res.json(parsed);

  } catch (err) {
    console.error("/analyze error:", err.message);
    res.json({
      whatThisReallyMeans: "The backend encountered an error.",
      impactLine: err.message,
      riskLevel: "Low",
      riskRead: "This is a technical error, not a real risk assessment.",
      whatToDo: ["Check Render logs", "Verify OPENAI_API_KEY is set", "Try again"],
      sayThis: { native: "There was an error. Please try again.", english: "There was an error. Please try again." },
      whatTheyWant: "System error.",
    });
  }
});

// ── /say — natural Vietnamese phrasing ────────────────────────────────────────

app.post("/say", async (req, res) => {
  try {
    const { text, tone } = req.body;

    const response = await openai.chat.completions.create({
      model: "gpt-4.1",
      messages: [
        {
          role: "system",
          content: `You are a cultural communication expert. You help people say things naturally in any language.

Your job is NOT to translate literally. Write what a real local person would actually say — idiomatically, with the right social register.

The user will describe what they want to say, in any language. Detect the language of their input and reply in THAT language. Never default to Vietnamese unless the input is in Vietnamese.

Return ONLY a valid JSON object. No markdown. No explanation outside the JSON.

{
  "native": "The message in the same language as the user's input. Natural phrasing for a real local speaker. Match the tone.",
  "english": "Plain English meaning. If native is already English, write a natural rephrasing.",
  "toneExplain": "One sentence describing how this sounds socially in that culture.",
  "variations": {
    "softer": "A softer version in the same language.",
    "direct": "A more direct version in the same language.",
    "shorter": "A shorter version in the same language."
  }
}

Tone guide:
- Polite: formal register, deferential but clear
- Casual: relaxed, natural between peers or friends
- Firm: direct, minimal softening, gets action

Context matters: dating, work, housing, family — adjust accordingly.
Sound like a person. Never robotic. Never textbook.`,
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
