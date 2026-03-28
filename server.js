console.log("TEST CHANGE");
import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

app.post("/analyze", (req, res) => {
  res.json({
    result: "TEST SUCCESS — backend is connected"
  });
});

app.listen(3000, "0.0.0.0", () => {
  console.log("Server running on port 3000");
});