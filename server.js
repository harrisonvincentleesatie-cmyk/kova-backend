console.log("TEST CHANGE");
import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

app.get("/", (req, res) => {
  res.send("Server is alive");
});

app.post("/analyze", (req, res) => {
  res.json({
    whatThisReallyMeans: "This is a test — backend is connected.",
    impactLine: "No real risk, just confirming connection.",
    riskLevel: "Low",
    riskRead: "This is just a test response.",
    whatToDo: [
      "Do nothing",
      "Confirm system works",
      "Proceed to next step"
    ],
    sayThis: "Test reply.",
    whatTheyWant: "Nothing — this is just a connection test."
  });
});

app.listen(3000, "0.0.0.0", () => {
  console.log("Server running on port 3000");
});