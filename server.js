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

    const systemPrompt = croppedImage
      ? `You are Kova. Analyse the images and return ONLY a valid JSON object — no markdown, no extra text.

You are given TWO images:
1. SELECTED MESSAGE — a cropped region the user explicitly chose. This is the ONLY message you must respond to.
2. FULL CONVERSATION — the complete screenshot. Use it only for tone, relationship, and context.

Treat the selected message as incoming — written by the other person to the user.
Generate a reply FROM the user back to the selected message.
Do not respond to any other message in the full conversation.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.`
      : `You are Kova. Analyse the screenshot and return ONLY a valid JSON object — no markdown, no extra text.

Focus on the message at approximately ${tapX}% from the left and ${tapY}% from the top.
Treat that message as incoming. Generate a reply from the user back to it.
Detect the conversation language and reply in that language. Never default to Vietnamese unless the screenshot is in Vietnamese.`;

    const userContent: any[] = [
      { type: "text", text: croppedImage ? "Selected message (image 1). Full conversation (image 2)." : "Analyse this screenshot." },
    ];
    if (croppedImage) {
      userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${croppedImage}` } });
    }
    userContent.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${image}` } });

    const jsonSchema = `{
  "whatThisReallyMeans": "Real intent behind the selected message. 1–2 sharp sentences.",
  "impactLine": "What happens if the user responds badly. One sentence.",
  "riskLevel": "Low" or "Medium" or "High",
  "riskRead": "One sentence on the risk level.",
  "whatToDo": ["Action", "Action", "Action"],
  "sayThis": {
    "native": "The user's reply to the selected message. In the conversation's language. Natural, not translated.",
    "english": "Plain English meaning. Rephrase if already English."
  },
  "whatTheyWant": "What the sender of the selected message wants. One sentence."
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
      whatThisReallyMeans: "Could not read the image clearly.",
      impactLine: "Try uploading a clearer screenshot.",
      riskLevel: "Low",
      riskRead: "Unable to assess — image may be unreadable.",
      whatToDo: ["Upload a clearer screenshot", "Ensure text is visible", "Try again"],
      sayThis: { native: "Unable to generate a reply.", english: "Unable to generate a reply." },
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
