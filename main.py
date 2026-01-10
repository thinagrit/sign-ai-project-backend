import React, { useEffect, useRef, useState } from "react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { 
  Loader2, Camera, Hand, Save, 
  Home as HomeIcon, Database, Activity, 
  CheckCircle, RefreshCw, ServerCrash,
  HeartPulse, Brain, BookOpen, 
  Video, MonitorCheck, StopCircle, WifiOff
} from "lucide-react";

// ==============================================
// ⚙️ ตั้งค่า API (URL สำหรับ Backend บน Render)
// ==============================================
const API_URL = "https://sign-ai-project-backend.onrender.com"; 
// ==============================================

// --- Components ---

function Navbar({ page, setPage }) {
  const menus = [
    { id: "home", label: "หน้าหลัก", icon: <HomeIcon size={20} /> },
    { id: "data", label: "สอนท่ามือ", icon: <Database size={20} /> },
    { id: "dictionary", label: "คลังคำศัพท์", icon: <BookOpen size={20} /> },
    { id: "predict", label: "แปลภาษา", icon: <Activity size={20} /> },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200/60 shadow-sm">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="flex justify-between items-center h-16">
          <div 
            className="flex items-center gap-2 font-bold text-2xl cursor-pointer"
            onClick={() => setPage("home")}
          >
            <div className="bg-gradient-to-tr from-blue-600 to-cyan-500 p-2 rounded-lg shadow-lg">
              <Hand className="text-white w-6 h-6" />
            </div>
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-slate-800 to-slate-600">
              ThaiMed<span className="font-light">AI</span>
            </span>
          </div>

          <div className="hidden md:flex bg-slate-100/50 p-1 rounded-xl">
            {menus.map((m) => (
              <button
                key={m.id}
                onClick={() => setPage(m.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 text-sm font-medium ${
                  page === m.id 
                    ? "bg-white text-blue-600 shadow-sm ring-1 ring-slate-200" 
                    : "text-slate-500 hover:text-slate-700 hover:bg-slate-200/50"
                }`}
              >
                {m.icon}
                <span className="hidden sm:inline">{m.label}</span>
              </button>
            ))}
          </div>

          <div className="md:hidden flex gap-2">
             {menus.map((m) => (
               <button
                 key={m.id}
                 onClick={() => setPage(m.id)}
                 className={`p-2 rounded-lg ${page === m.id ? "bg-blue-100 text-blue-600" : "text-slate-500"}`}
               >
                 {m.icon}
               </button>
             ))}
          </div>
        </div>
      </div>
    </nav>
  );
}

function HomePage({ setPage }) {
  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col justify-center items-center px-4 relative overflow-hidden">
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-200/20 rounded-full blur-3xl -z-10 animate-pulse" />
      <div className="max-w-4xl w-full text-center space-y-12">
        <div className="space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-50 border border-blue-100 text-blue-600 text-sm font-medium shadow-sm">
            <HeartPulse size={16} /> นวัตกรรมเพื่อการแพทย์ยุคใหม่
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-slate-900 leading-tight">
            ระบบแปลภาษามือ<br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-500">
              ด้วยปัญญาประดิษฐ์
            </span>
          </h1>
          <p className="text-xl text-slate-500 max-w-2xl mx-auto font-light leading-relaxed">
            เชื่อมต่อการสื่อสารระหว่างบุคลากรทางการแพทย์และผู้บกพร่องทางการได้ยิน
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <button onClick={() => setPage("data")} className="group p-6 bg-white rounded-3xl border border-slate-100 shadow-xl hover:-translate-y-1 transition-all text-left">
            <div className="h-12 w-12 bg-blue-50 rounded-xl flex items-center justify-center text-blue-600 mb-4">
              <Database size={24} />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-1">สอนท่ามือ</h3>
            <p className="text-slate-500 text-sm">บันทึก 60 เฟรมต่อเนื่อง</p>
          </button>

          <button onClick={() => setPage("dictionary")} className="group p-6 bg-white rounded-3xl border border-slate-100 shadow-xl hover:-translate-y-1 transition-all text-left">
            <div className="h-12 w-12 bg-purple-50 rounded-xl flex items-center justify-center text-purple-600 mb-4">
              <BookOpen size={24} />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-1">คลังคำศัพท์</h3>
            <p className="text-slate-500 text-sm">ดูรายการคำศัพท์ที่มีอยู่ใน Dataset</p>
          </button>

          <button onClick={() => setPage("predict")} className="group p-6 bg-slate-900 rounded-3xl shadow-xl hover:-translate-y-1 transition-all text-left">
            <div className="h-12 w-12 bg-white/10 rounded-xl flex items-center justify-center text-cyan-400 mb-4">
              <Activity size={24} />
            </div>
            <h3 className="text-xl font-bold text-white mb-1">แปลภาษา</h3>
            <p className="text-slate-400 text-sm">แปลผลแบบ Sliding Window</p>
          </button>
        </div>
      </div>
    </div>
  );
}

function DictionaryPage() {
  const [stats, setStats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchData = async () => {
        try {
            const res = await fetch(`${API_URL}/dataset`);
            if (!res.ok) throw new Error("Failed to connect");
            const data = await res.json();
            if (isMounted) {
                const grouped = data.reduce((acc, curr) => {
                    acc[curr.label] = (acc[curr.label] || 0) + 1;
                    return acc;
                }, {});
                setStats(Object.entries(grouped).map(([label, count]) => ({ label, count })));
                setLoading(false);
            }
        } catch (err) {
            if (isMounted) {
                setError("เชื่อมต่อไม่ได้ (ตรวจสอบการตั้งค่า Backend)");
                setLoading(false);
            }
        }
    };
    fetchData();
    return () => { isMounted = false; };
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <h2 className="text-3xl font-bold mb-8 flex items-center gap-3">
        <BookOpen className="text-purple-500" /> คลังคำศัพท์ในระบบ
      </h2>
      {loading ? <div className="text-center py-20"><Loader2 className="animate-spin mx-auto w-10 h-10 text-blue-500" /></div> : 
       error ? <div className="bg-red-50 text-red-500 p-10 text-center rounded-3xl border border-red-100">{error}</div> :
       <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
         {stats.map((s, i) => (
           <div key={i} className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:border-purple-200 transition-colors">
             <h3 className="text-xl font-bold text-slate-800">{s.label}</h3>
             <p className="text-slate-400 font-medium">{s.count} Sequences</p>
           </div>
         ))}
       </div>
      }
    </div>
  );
}

// --- MediaPipe Logic ---
const useHandTracking = (videoRef, onResults) => {
  const [loading, setLoading] = useState(true);
  const landmarkerRef = useRef(null);

  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });
        if(isMounted) {
          landmarkerRef.current = landmarker;
          setLoading(false);
        }
      } catch(e) { console.error("MediaPipe Error:", e); }
    })();
    return () => { isMounted = false; };
  }, []);

  const processVideo = () => {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;

    if (video && landmarker && video.readyState >= 2) {
      // ✅ ส่ง imageWidth/Height เพื่อกำจัด OpenGL Warning
      const results = landmarker.detectForVideo(
        video, 
        performance.now(),
        {
          imageWidth: video.videoWidth,
          imageHeight: video.videoHeight,
        }
      );
      onResults(results);
    }
    requestAnimationFrame(processVideo);
  };
  return { loading, startLoop: processVideo };
};

function CameraView({ onLandmarks, overlayText, showSkeleton = true }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);

  const { loading, startLoop } = useHandTracking(videoRef, (results) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);

    if (results.landmarks && results.landmarks.length > 0) {
      if (showSkeleton) {
        results.landmarks.forEach(hand => {
          hand.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, 2 * Math.PI);
            ctx.fillStyle = "#3B82F6";
            ctx.fill();
            ctx.strokeStyle = "white";
            ctx.lineWidth = 1;
            ctx.stroke();
          });
        });
      }
      // ดึงข้อมูล 21 จุด (x, y, z)
      const flatPoints = results.landmarks.flatMap(hand => hand.flatMap(p => [p.x, p.y, p.z]));
      onLandmarks(flatPoints);
    } else {
      onLandmarks(null);
    }
    ctx.restore();
  });

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720, facingMode: "user" } 
      });
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play();
        setCameraActive(true);
        startLoop();
      };
    } catch (e) { alert("ไม่สามารถเข้าถึงกล้องได้"); }
  };

  return (
    <div className="relative rounded-3xl overflow-hidden bg-slate-900 aspect-video shadow-2xl border border-slate-800 ring-4 ring-slate-100">
      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-white z-20 bg-slate-900">
          <Loader2 className="animate-spin mb-4 text-blue-500 w-10 h-10" />
          <p className="font-medium animate-pulse">กำลังดาวน์โหลด AI Hand Engine...</p>
        </div>
      )}
      {!loading && !cameraActive && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-slate-900/40 backdrop-blur-sm">
          <button onClick={startCamera} className="px-8 py-4 bg-white text-slate-900 rounded-full font-bold shadow-2xl flex items-center gap-3 hover:scale-105 transition-transform active:scale-95">
            <Camera size={24} className="text-blue-600" /> <span>เริ่มต้นเปิดกล้อง</span>
          </button>
        </div>
      )}
      <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]" playsInline muted />
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]" />
      {overlayText && cameraActive && (
        <div className="absolute top-6 left-1/2 -translate-x-1/2 z-30">
           <div className="bg-slate-900/80 text-white px-6 py-2 rounded-full border border-white/20 text-sm font-medium backdrop-blur-md shadow-xl flex items-center gap-2">
             <div className="w-2 h-2 bg-red-500 rounded-full animate-ping" />
             {overlayText}
           </div>
        </div>
      )}
    </div>
  );
}

function DataPage() {
  const [currentLandmarks, setCurrentLandmarks] = useState(null);
  const [label, setLabel] = useState("");
  const [status, setStatus] = useState("ready");
  const [progress, setProgress] = useState(0);
  const framesBuffer = useRef([]);

  const handleLandmarksUpdate = (points) => {
    setCurrentLandmarks(points);
    if (status === "recording" && points) {
        framesBuffer.current.push(points);
        setProgress(framesBuffer.current.length);
        if (framesBuffer.current.length >= 60) finishRecording();
    }
  };

  const startRecording = () => {
    if (!label.trim()) return alert("กรุณาระบุชื่อท่ามือที่ต้องการสอน");
    if (!currentLandmarks) return alert("AI ยังตรวจจับมือไม่เจอ กรุณายกมือขึ้นครับ");
    framesBuffer.current = [];
    setProgress(0);
    setStatus("recording");
  };

  const finishRecording = async () => {
    setStatus("uploading");
    try {
      // ✅ เปลี่ยนเป็น /upload-sequence และ Schema: label, frames
      const res = await fetch(`${API_URL}/upload-sequence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
            label: label.trim(), 
            frames: framesBuffer.current 
        })
      });
      if(!res.ok) throw new Error("ไม่สามารถส่งข้อมูลได้ (Check API Endpoint)");
      setStatus("success");
      setTimeout(() => setStatus("ready"), 2000);
      setLabel(""); setProgress(0);
    } catch (e) {
      alert(`Error: ${e.message}`);
      setStatus("ready");
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 flex flex-col lg:flex-row gap-10">
      <div className="flex-1">
        <CameraView onLandmarks={handleLandmarksUpdate} overlayText={status === "recording" ? `กำลังบันทึกเฟรม: ${progress}/60` : "พร้อมบันทึก (ต้องการ 60 เฟรม)"} />
      </div>
      <div className="lg:w-96 space-y-6">
        <div className="bg-white p-6 rounded-3xl border shadow-sm">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2"><Database size={20} className="text-blue-500"/> สอนท่าทางใหม่</h2>
            <div className="space-y-4">
                <input 
                    type="text" 
                    value={label} 
                    onChange={e => setLabel(e.target.value)} 
                    placeholder="เช่น ปวดหัว, ไม่สบาย..." 
                    className="w-full p-4 border rounded-2xl outline-none focus:ring-2 focus:ring-blue-500 transition-all" 
                />
                <button 
                    onClick={startRecording} 
                    disabled={status !== "ready"}
                    className={`w-full py-4 rounded-2xl font-bold transition-all shadow-lg ${
                        status === "recording" ? "bg-red-500 text-white animate-pulse" : 
                        status === "uploading" ? "bg-slate-300 text-slate-500 cursor-not-allowed" :
                        "bg-blue-600 text-white hover:bg-blue-700 active:scale-95 shadow-blue-500/20"
                    }`}
                >
                    {status === "recording" ? `กำลังบันทึกข้อมูล (${progress}/60)` : 
                     status === "uploading" ? "กำลังประมวลผล..." : 
                     status === "success" ? "บันทึกสำเร็จ!" : "เริ่มการสอนท่าทาง"}
                </button>
                <p className="text-xs text-slate-400 text-center">รวบรวม 60 เฟรมเพื่อความแม่นยำในการวิเคราะห์</p>
            </div>
        </div>
      </div>
    </div>
  );
}

function PredictPage() {
  const [result, setResult] = useState({ label: "รอการตรวจจับ...", confidence: 0 });
  const [error, setError] = useState(false);
  const [bufferCount, setBufferCount] = useState(0);
  const framesRef = useRef([]); // สำหรับสะสม 60 เฟรม
  const lastSentRef = useRef(0);

  const handleLandmarks = async (points) => {
    if (!points) {
        // ถ้าไม่เจอจุด ให้ล้าง Buffer บางส่วนเพื่อป้องกันข้อมูลขาดตอน
        if (framesRef.current.length > 0) framesRef.current.shift();
        setBufferCount(framesRef.current.length);
        return;
    }

    // ✅ เพิ่มข้อมูลลงใน Sliding Window Buffer
    framesRef.current.push(points);
    if (framesRef.current.length > 60) {
        framesRef.current.shift(); // ลบเฟรมเก่าที่สุดออก
    }
    setBufferCount(framesRef.current.length);

    // ต้องรอให้ครบ 60 เฟรมก่อนเริ่มส่ง Predict
    if (framesRef.current.length < 60) return;

    // หน่วงเวลาการส่ง เพื่อไม่ให้ Server ทำงานหนักเกินไป (ทุก 300ms)
    const now = Date.now();
    if (now - lastSentRef.current < 300) return;
    lastSentRef.current = now;

    try {
      // ✅ เปลี่ยนเป็น /predict-sequence และ Schema: frames
      const res = await fetch(`${API_URL}/predict-sequence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames: framesRef.current }) 
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
        setError(false);
      } else { setError(true); }
    } catch (e) { setError(true); }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
      {error && <div className="bg-red-50 text-red-500 p-4 rounded-2xl text-center border border-red-100 flex items-center justify-center gap-2">
        <WifiOff size={18}/> ขัดข้อง: กรุณาตรวจสอบสถานะของ Server (Render)
      </div>}
      
      <CameraView 
        onLandmarks={handleLandmarks} 
        overlayText={bufferCount < 60 ? `กำลังรวบรวมท่าทาง (${bufferCount}/60)...` : "พร้อมแปลผล"} 
      />

      <div className="p-10 bg-white rounded-[2.5rem] shadow-xl border text-center relative overflow-hidden group">
        <div className="absolute top-0 left-0 w-full h-1 bg-slate-50 overflow-hidden">
            <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${result.confidence * 100}%` }} />
        </div>
        
        <p className="text-slate-400 text-sm font-semibold uppercase tracking-widest mb-2">คำแปลที่ได้</p>
        <h2 className={`text-7xl font-black mb-6 transition-all ${result.confidence > 0.6 ? "text-blue-600 scale-105" : "text-slate-300"}`}>
          {result.label}
        </h2>

        <div className="flex flex-col items-center gap-2">
            <div className="flex items-center gap-3">
                <span className="text-xs font-bold text-slate-400">ระดับความมั่นใจ</span>
                <span className={`text-sm font-mono font-bold px-3 py-1 rounded-full ${result.confidence > 0.6 ? "bg-green-100 text-green-600" : "bg-slate-100 text-slate-500"}`}>
                    {(result.confidence * 100).toFixed(1)}%
                </span>
            </div>
            <p className="text-[10px] text-slate-300 uppercase tracking-tighter italic">Sequence Processing Enabled</p>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("home");
  return (
    <div className="min-h-screen bg-slate-50/50 pt-16 selection:bg-blue-100 selection:text-blue-900">
      <Navbar page={page} setPage={setPage} />
      <main className="container mx-auto">
        {page === "home" && <HomePage setPage={setPage} />}
        {page === "data" && <DataPage />}
        {page === "dictionary" && <DictionaryPage />}
        {page === "predict" && <PredictPage />}
      </main>
    </div>
  );
}
