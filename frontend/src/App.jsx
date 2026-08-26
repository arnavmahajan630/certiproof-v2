import { useEffect, useState } from "react";
import { api } from "./api/client.js";
import LandingPage from "./pages/LandingPage.jsx";
import Student from "./views/Student.jsx";
import Teacher from "./views/Teacher.jsx";
import Auditor from "./views/Auditor.jsx";
import "./App.css";

const ROLE_LABEL = { student: "Student", teacher: "Teacher", auditor: "Auditor" };

function App() {
  const [view, setView] = useState("landing");
  const [apiUp, setApiUp] = useState(null);

  useEffect(() => {
    api.health().then(() => setApiUp(true)).catch(() => setApiUp(false));
  }, []);

  if (view === "landing") {
    return (
      <div className="app">
        <LandingPage onSelectRole={setView} />
      </div>
    );
  }

  return (
    <div className="app">
      <div className="topbar">
        <div className="topbar-left">
          <span className="topbar-title">CertiProof</span>
          {view !== "student" && <span className="topbar-role">{ROLE_LABEL[view]} view</span>}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {view !== "student" && (
            <span className={`status inline ${apiUp ? "ok" : "fail"}`}>{apiUp ? "Backend online" : "Backend offline"}</span>
          )}
          <button className="back-link" onClick={() => setView("landing")}>
            ← Back to role selection
          </button>
        </div>
      </div>

      <main className={`main${view === "student" ? " student-main" : ""}`}>
        {view === "student" && <Student />}
        {view === "teacher" && <Teacher />}
        {view === "auditor" && <Auditor />}
      </main>
    </div>
  );
}

export default App;
