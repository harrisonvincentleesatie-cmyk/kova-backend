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

app.post("/analyze", async (req, res) => {
  try {
    const { image } = req.body;

    const response = await openai.chat.completions.create({
      model: "gpt-4.1",
      messages: [
        {
          role: "system",
          content: `
You are Kova — an AI that reads between the lines of real-world communication.

Follow this structure EXACTLY:

1. WHAT THIS REALLY MEANS
2. RISK
3. WHAT TO DO
4. SAY THIS
5. WHAT THEY’RE TRYING TO DO

Be sharp, concise, and confident.
No generic language.
`,
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Analyze this screenshot of a message.",
            },
            {
              type: "image_url",
              image_url: {
                url: image,
              },
            },
          ],
        },
      ],
    });

    res.json({ result: response.choices[0].message.content });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(3000, '0.0.0.0', () => {
  console.log("Server running on port 3000");
});