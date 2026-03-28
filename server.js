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
          content: `You are Kova. You read between the lines of real messages and tell people what is actually happening and exactly what to say.

Return ONLY a valid JSON object. No markdown, no code blocks, no extra text.

{
  "whatThisReallyMeans": "1–2 sentences. What is actually happening — the real intent, not the surface words. Commit to one interpretation.",
  "impactLine": "One sentence. What happens if they respond badly.",
  "riskLevel": "Low" or "Medium" or "High",
  "riskRead": "One sentence explaining the risk level.",
  "whatToDo": ["Short action", "Short action", "Short action"],
  "sayThis": {
    "vietnamese": "The reply they should send. Written in natural, conversational Vietnamese — how a real local person would say it. Match the register: formal for landlord or work, relaxed for friends.",
    "english": "What the Vietnamese reply means in plain English."
  },
  "whatTheyWant": "One sentence. The real intent behind their message."
}

Rules:
- Only describe relationships that are clearly visible (do not assume manager, HR, or authority)
- No corporate language: escalate, loop in, circle back, touch base
- The Vietnamese reply must sound like a real person, not a textbook
- Tone by situation: landlord = polite but firm / work = calm and direct / casual = natural
- No dramatic language, no invented urgency
- Always commit to one clear read`,
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
      sayThis: { vietnamese: "Đã xảy ra lỗi. Vui lòng thử lại.", english: "There was an error. Please try again." },
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
