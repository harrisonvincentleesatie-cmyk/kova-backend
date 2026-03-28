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

app.get("/", (req, res) => {
  res.send("Server is alive");
});

app.post("/analyze", async (req, res) => {
  try {
    const { image } = req.body;

    const response = await openai.chat.completions.create({
      model: "gpt-4.1",
      messages: [
        {
          role: "system",
          content: `You are Kova — an AI that reads between the lines of real-world communication.

Analyse the screenshot and respond with ONLY a valid JSON object in this exact structure, no extra text:

{
  "whatThisReallyMeans": "1-2 sharp sentences. What is actually happening, not what was said.",
  "impactLine": "One sentence on the personal consequence if the user responds wrong.",
  "riskLevel": "Low" or "Medium" or "High",
  "riskRead": "One short sentence explaining the risk level.",
  "whatToDo": ["Action 1", "Action 2", "Action 3"],
  "sayThis": "The exact reply the user should send.",
  "whatTheyWant": "One sentence psychological read on the sender's intent."
}

Rules:
- Commit to one interpretation. No hedging.
- Be concise, intelligent, slightly ruthless.
- Never use generic language.
- Return only the JSON. No markdown, no code blocks, no explanation.`,
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

    const raw = response.choices[0].message.content.trim();

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (parseErr) {
      console.error("Failed to parse OpenAI response as JSON:", raw);
      return res.status(500).json({ error: "OpenAI returned unexpected format" });
    }

    res.json(parsed);

  } catch (err) {
    console.error("OpenAI call failed:", err.message);
    res.status(500).json({ error: "OpenAI call failed: " + err.message });
  }
});

app.listen(3000, "0.0.0.0", () => {
  console.log("Server running on port 3000");
});
