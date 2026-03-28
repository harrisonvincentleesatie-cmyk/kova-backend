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
  res.json({
    result: "TEST SUCCESS — backend is connected"
  });
});

app.listen(3000, '0.0.0.0', () => {
  console.log("Server running on port 3000");
});