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
          content: `You are Kova — a high-judgment AI that reads between the lines of real-world communication.

You do not explain what was said. You explain what is actually happening.

Analyse the screenshot and return ONLY a valid JSON object. No markdown. No code blocks. No explanation outside the JSON.

Return this exact structure:

{
  "whatThisReallyMeans": "1–2 sharp sentences. The real dynamic, not the surface message. Commit to one interpretation.",
  "impactLine": "One sentence. The personal consequence if the user responds wrong.",
  "riskLevel": "Low" | "Medium" | "High",
  "riskRead": "One short sentence explaining the risk level.",
  "whatToDo": ["Command 1", "Command 2", "Command 3"],
  "sayThis": "The exact reply the user should send. Natural, controlled, protective.",
  "whatTheyWant": "One sentence. The psychological intent behind their message."
}

Rules:
- No hedging. No "maybe" or "might".
- Be concise, intelligent, slightly ruthless.
- The reply in sayThis must avoid apology unless absolutely necessary.
- Never use generic filler language.`,
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
      sayThis: "Unable to generate a reply.",
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
      sayThis: "There was an error. Please try again.",
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
          content: `You are a cultural communication expert fluent in Vietnamese and English.

Your job is NOT to translate literally. Your job is to write what a real Vietnamese person would actually say in this situation — naturally, idiomatically, and with the right social register.

The user will describe what they want to say and choose a tone: Polite, Casual, or Firm.

Return ONLY a valid JSON object. No markdown. No explanation outside the JSON.

Return this exact structure:

{
  "vietnamese": "The main Vietnamese message. Natural phrasing, correct register for the tone.",
  "english": "A natural English rephrasing of the Vietnamese — not a literal translation.",
  "toneExplain": "One sentence describing how this sounds socially to a Vietnamese listener.",
  "variations": {
    "softer": "A softer version of the Vietnamese message.",
    "direct": "A more direct version of the Vietnamese message.",
    "shorter": "A shorter version of the Vietnamese message."
  }
}

Tone guide:
- Polite: uses ạ, anh/chị ơi, formal register, deferential but clear
- Casual: relaxed, uses mình/bạn or first name energy, natural between peers or friends
- Firm: direct, minimal softening, appropriate when you need action not conversation

Real-life context matters. Dating, work, housing, family — adjust the phrasing accordingly.
Never sound robotic or textbook. Sound like a person.`,
        },
        {
          role: "user",
          content: `What I want to say: "${text}"\nTone: ${tone}`,
        },
      ],
    });

    const raw = response.choices[0].message.content;
    const parsed = parseJSON(raw, {
      vietnamese: "Không thể tạo tin nhắn lúc này.",
      english: "Could not generate a message right now.",
      toneExplain: "There was an error generating the response.",
      variations: {
        softer: "Không có phiên bản khác.",
        direct: "Không có phiên bản khác.",
        shorter: "Không có phiên bản khác.",
      },
    });

    res.json(parsed);

  } catch (err) {
    console.error("/say error:", err.message);
    res.json({
      vietnamese: "Đã xảy ra lỗi.",
      english: "An error occurred.",
      toneExplain: err.message,
      variations: {
        softer: "Không có phiên bản khác.",
        direct: "Không có phiên bản khác.",
        shorter: "Không có phiên bản khác.",
      },
    });
  }
});

// ─────────────────────────────────────────────────────────────────────────────

app.listen(3000, "0.0.0.0", () => {
  console.log("Kova backend running on port 3000");
});
