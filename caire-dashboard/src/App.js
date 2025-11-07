import "./App.css";
import { useState, useEffect, useRef } from "react";
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  Container,
  ThemeProvider,
  createTheme,
  Button,
} from "@mui/material";
import {
  Favorite,
  Opacity,
  Air,
  Thermostat,
  Psychology,
  MonitorHeart,
} from "@mui/icons-material";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#00ff88",
    },
    secondary: {
      main: "#00d9ff",
    },
    background: {
      default: "#0a0e1a",
      paper: "#12161f",
    },
    success: {
      main: "#00ff88",
    },
    warning: {
      main: "#ffaa00",
    },
    error: {
      main: "#ff3366",
    },
    text: {
      primary: "#ffffff",
      secondary: "#8899aa",
    },
  },
  typography: {
    fontFamily:
      '"Orbitron", "Rajdhani", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
  },
});

function App() {
  const [healthMetrics, setHealthMetrics] = useState({
    heartRate: null, // Will be populated from API
    bloodPressure: { systolic: 120, diastolic: 80 }, // Dummy
    oxygenLevel: 98, // Dummy
    respiratoryRate: 16, // Dummy
    temperature: 98.6, // Dummy
    stressLevel: "Low", // Dummy
  });

  const [apiStatus, setApiStatus] = useState({
    camera: "idle", // idle, initializing, connected, stopped, error
    websocket: "idle", // idle, initializing, connected, streaming, finished, error
    isStreaming: false,
    framesSent: 0,
  });

  const [showVideoPreview, setShowVideoPreview] = useState(false);
  const [isStarted, setIsStarted] = useState(false);
  const [heartRateWarning, setHeartRateWarning] = useState(null); // null, "low", "high"

  const [rppgData, setRppgData] = useState([]);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const wsRef = useRef(null);
  const loopTimerRef = useRef(null);

  const frameCountRef = useRef(0);

  // Constants
  const WS_BASE = "ws://3.67.186.245:8003/ws/";
  const API_KEY = "ZzTf4iMeWmMq-wifnlT3sAjyIZba6FYtF8DoDrvTfcQ";
  const FPS = 30;
  const ADVANCED = true;

  // Helper function to get JPEG base64 without prefix
  const getJpegBase64NoPrefix = (video, canvas, quality = 0.9) => {
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", quality);
    return dataUrl.replace(/^data:image\/jpeg;base64,/, "");
  };

  // Build payload for API
  const buildPayload = (state, videoEl, canvasEl, advancedFlag) => {
    return {
      datapt_id: crypto.randomUUID(),
      state,
      frame_data: getJpegBase64NoPrefix(videoEl, canvasEl),
      timestamp: (Date.now() / 1000).toString(),
      advanced: advancedFlag,
    };
  };

  // Build WebSocket URL
  const buildWsUrl = (base, apiKey) => {
    const urlBase = base.replace(/\/+$/, "");
    const qs = new URLSearchParams({ api_key: apiKey }).toString();
    return `${urlBase}/?${qs}`;
  };

  // Start streaming function
  const startStreaming = async () => {
    try {
      console.log("🎥 Starting camera and WebSocket connection...");
      setApiStatus((prev) => ({ ...prev, camera: "initializing" }));
      setIsStarted(true);
      frameCountRef.current = 0;

      // Get camera stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: FPS, max: FPS },
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        console.log("✅ Camera started");
        setApiStatus((prev) => ({ ...prev, camera: "connected" }));
        setShowVideoPreview(true);
      }

      // Connect to WebSocket
      const wsUrl = buildWsUrl(WS_BASE, API_KEY);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      setApiStatus((prev) => ({ ...prev, websocket: "initializing" }));

      ws.onopen = () => {
        console.log("✅ WebSocket connected");
        setApiStatus((prev) => ({
          ...prev,
          websocket: "connected",
          isStreaming: true,
        }));

        // Start sending frames continuously
        const tickMs = Math.max(1, Math.floor(1000 / Math.max(1, FPS)));
        loopTimerRef.current = setInterval(() => {
          if (!ws || ws.readyState >= 2) {
            clearInterval(loopTimerRef.current);
            return;
          }

          if (videoRef.current && captureCanvasRef.current) {
            const payload = buildPayload(
              "stream",
              videoRef.current,
              captureCanvasRef.current,
              ADVANCED
            );
            ws.send(JSON.stringify(payload));
            frameCountRef.current++;

            setApiStatus((prev) => ({
              ...prev,
              websocket: "streaming",
              framesSent: frameCountRef.current,
            }));
          }
        }, tickMs);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("📥 Server response:", data);

          // Update health metrics from API response (live updates)
          if (data.inference && data.inference.hr) {
            const heartRate = Math.round(data.inference.hr);
            setHealthMetrics((prev) => ({
              ...prev,
              heartRate: heartRate,
            }));

            // Check for heart rate warnings (exclude invalid values like -1)
            if (heartRate > 0 && heartRate < 60) {
              setHeartRateWarning("low");
            } else if (heartRate > 100) {
              setHeartRateWarning("high");
            } else {
              setHeartRateWarning(null);
            }
          }

          // Update rPPG waveform from advanced data (live updates)
          if (
            data.advanced &&
            data.advanced.rppg &&
            Array.isArray(data.advanced.rppg)
          ) {
            setRppgData(data.advanced.rppg.slice(-100));
          }
        } catch (error) {
          console.log("📥 Server response (raw):", event.data);
        }
      };

      ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        setApiStatus((prev) => ({
          ...prev,
          websocket: "error",
          isStreaming: false,
        }));
      };

      ws.onclose = (event) => {
        console.log("🔌 WebSocket closed:", event.code, event.reason || "");
        if (loopTimerRef.current) {
          clearInterval(loopTimerRef.current);
        }
        setApiStatus((prev) => ({
          ...prev,
          isStreaming: false,
        }));
      };
    } catch (error) {
      console.error("❌ Failed to start streaming:", error);
      setApiStatus((prev) => ({
        ...prev,
        camera: "error",
        websocket: "error",
        isStreaming: false,
      }));
      setIsStarted(false);
    }
  };

  // Stop streaming function
  const stopStreaming = () => {
    console.log("🛑 Stopping streaming...");

    // Send final "end" payload
    if (
      wsRef.current &&
      wsRef.current.readyState === 1 &&
      videoRef.current &&
      captureCanvasRef.current
    ) {
      const finalPayload = buildPayload(
        "end",
        videoRef.current,
        captureCanvasRef.current,
        ADVANCED
      );
      wsRef.current.send(JSON.stringify(finalPayload));
      console.log("🏁 Sent final frame");
    }

    // Clear interval
    if (loopTimerRef.current) {
      clearInterval(loopTimerRef.current);
      loopTimerRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current && wsRef.current.readyState < 2) {
      wsRef.current.close();
    }

    // Stop camera
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      console.log("📹 Camera stopped");
    }

    setShowVideoPreview(false);
    setApiStatus((prev) => ({
      ...prev,
      isStreaming: false,
      camera: "stopped",
      websocket: "stopped",
    }));
    setIsStarted(false);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (loopTimerRef.current) {
        clearInterval(loopTimerRef.current);
      }
      if (wsRef.current && wsRef.current.readyState < 2) {
        wsRef.current.close();
      }
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  // Draw waveform on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas with transparent background
    ctx.clearRect(0, 0, width, height);

    // Draw subtle background grid
    ctx.strokeStyle = "#00ff8820";
    ctx.lineWidth = 1;

    // Draw horizontal grid lines
    const horizontalSpacing = height / 6;
    for (let y = 0; y <= height; y += horizontalSpacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw vertical grid lines
    const verticalSpacing = 50;
    for (let x = 0; x < width; x += verticalSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Draw waveform with glow effect
    if (rppgData.length > 1) {
      // Calculate dynamic scaling based on data range
      const minValue = Math.min(...rppgData);
      const maxValue = Math.max(...rppgData);
      const dataRange = maxValue - minValue;

      // Avoid division by zero and ensure minimum visibility
      const effectiveRange = dataRange > 0.001 ? dataRange : 1;

      // Use 80% of canvas height for the waveform (leave 10% padding top/bottom)
      const availableHeight = height * 0.8;
      const scaleFactor = availableHeight / effectiveRange;

      // Center point for the waveform
      const midValue = (maxValue + minValue) / 2;

      // Function to scale and center the waveform
      const scaleY = (value) => {
        // Normalize value relative to center, apply scale, then position on canvas
        const normalized = (value - midValue) * scaleFactor;
        return height / 2 - normalized; // Invert Y (canvas origin is top-left)
      };

      // Draw glowing background line
      ctx.strokeStyle = "#00ff8840";
      ctx.lineWidth = 8;
      ctx.beginPath();

      const step = width / rppgData.length;
      rppgData.forEach((value, index) => {
        const x = index * step;
        const y = scaleY(value);

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Draw main bright line
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 3;
      ctx.beginPath();

      rppgData.forEach((value, index) => {
        const x = index * step;
        const y = scaleY(value);

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }
  }, [rppgData]);

  // Reusable Alert Modal Component
  const AlertModal = ({
    show,
    type = "warning", // "warning", "info", "error", "success"
    title,
    message,
    position = { top: 20, right: 20 }, // Customizable position
    color = "#ff3366",
  }) => {
    if (!show) return null;

    return (
      <Box
        sx={{
          position: "fixed",
          top: position.top,
          right: position.right,
          left: position.left,
          bottom: position.bottom,
          minWidth: 300,
          maxWidth: 400,
          borderRadius: 2,
          overflow: "hidden",
          border: `2px solid ${color}`,
          boxShadow: `0 0 30px ${color}60`,
          bgcolor: `${color}20`,
          zIndex: 1000,
          animation: "pulseAlert 1.5s infinite",
          "@keyframes pulseAlert": {
            "0%, 100%": {
              opacity: 1,
              boxShadow: `0 0 30px ${color}60`,
            },
            "50%": {
              opacity: 0.8,
              boxShadow: `0 0 50px ${color}aa`,
            },
          },
        }}
      >
        <Box sx={{ p: 2.5 }}>
          <Typography
            variant="h6"
            sx={{
              color: color,
              fontFamily: '"Orbitron", monospace',
              fontWeight: 700,
              letterSpacing: 1.5,
              textShadow: `0 0 15px ${color}`,
              mb: 1.5,
              textAlign: "center",
              fontSize: "0.95rem",
            }}
          >
            {title}
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: "#ffffff",
              fontFamily: '"Rajdhani", sans-serif',
              fontWeight: 600,
              textAlign: "center",
              fontSize: "0.9rem",
              lineHeight: 1.5,
            }}
          >
            {message}
          </Typography>
        </Box>
      </Box>
    );
  };

  const MetricCard = ({
    icon: Icon,
    title,
    value,
    unit,
    status,
    color,
    isRealData = false,
  }) => (
    <Card
      elevation={0}
      sx={{
        height: "100%",
        bgcolor: "transparent",
        border: "none",
      }}
    >
      <CardContent
        sx={{
          p: 0,
          "&:last-child": { pb: 0 },
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
        }}
      >
        <Box display="flex" alignItems="center" gap={1.5}>
          <Box
            sx={{
              bgcolor: `${color || "#00ff88"}20`,
              borderRadius: 1.5,
              p: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "2px solid",
              borderColor: color || "primary.main",
            }}
          >
            <Icon sx={{ color: color || "primary.main", fontSize: 28 }} />
          </Box>
          <Box flex={1}>
            <Typography
              variant="caption"
              sx={{
                color: "#8899aa",
                textTransform: "uppercase",
                fontWeight: 600,
                letterSpacing: 1,
                fontSize: "0.65rem",
              }}
            >
              {title}
            </Typography>
            <Box display="flex" alignItems="baseline" gap={0.5}>
              <Typography
                variant="h4"
                fontWeight={700}
                sx={{
                  lineHeight: 1,
                  color: color || "primary.main",
                  fontFamily: '"Orbitron", monospace',
                  textShadow: `0 0 10px ${color || "#00ff88"}80`,
                }}
              >
                {value}
              </Typography>
              <Typography
                variant="caption"
                sx={{ color: "#8899aa", fontSize: "0.7rem" }}
              >
                {unit}
              </Typography>
            </Box>
          </Box>
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              bgcolor: isRealData
                ? status === "normal"
                  ? "#00ff88"
                  : "#ffaa00"
                : "#ff3366",
              boxShadow: isRealData
                ? status === "normal"
                  ? "0 0 12px #00ff88"
                  : "0 0 12px #ffaa00"
                : "0 0 12px #ff3366",
              animation: status === "normal" ? "none" : "pulse 2s infinite",
            }}
          />
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          height: "100vh",
          background:
            "radial-gradient(circle at 20% 50%, #0a1628 0%, #000510 100%)",
          p: 3,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <Box textAlign="center" color="white" mb={3}>
          <Typography
            variant="h3"
            fontWeight={700}
            sx={{
              textShadow: "0 0 20px #00ff88",
              color: "#00ff88",
              fontFamily: '"Orbitron", monospace',
              letterSpacing: 3,
            }}
          >
            DRIVER VITALS
          </Typography>
          <Typography
            variant="body2"
            sx={{ color: "#00d9ff", letterSpacing: 2, mt: 0.5 }}
          >
            REAL-TIME DRIVER ANALYSIS SYSTEM FOR CAIRE
          </Typography>
          {/* Status Indicator and Control Buttons */}
          <Box
            sx={{
              mt: 2,
              display: "flex",
              gap: 3,
              justifyContent: "center",
              alignItems: "center",
              flexWrap: "wrap",
            }}
          >
            {/* Status Indicators */}
            <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
              <Typography
                variant="caption"
                sx={{ color: "#8899aa", fontSize: "0.75rem" }}
              >
                Camera:{" "}
                {apiStatus.camera === "connected"
                  ? "✅"
                  : apiStatus.camera === "initializing"
                  ? "⏳"
                  : apiStatus.camera === "idle"
                  ? "⚪"
                  : apiStatus.camera === "stopped"
                  ? "⏹️"
                  : "❌"}
              </Typography>
              <Typography
                variant="caption"
                sx={{ color: "#8899aa", fontSize: "0.75rem" }}
              >
                WebSocket:{" "}
                {apiStatus.websocket === "streaming"
                  ? "🔄 Streaming"
                  : apiStatus.websocket === "connected"
                  ? "✅ Connected"
                  : apiStatus.websocket === "stopped"
                  ? "⏹️ Stopped"
                  : apiStatus.websocket === "initializing"
                  ? "⏳ Connecting"
                  : apiStatus.websocket === "idle"
                  ? "⚪ Idle"
                  : "❌ Error"}
              </Typography>
            </Box>

            {/* Control Buttons */}
            <Box sx={{ display: "flex", gap: 2 }}>
              <Button
                variant="contained"
                onClick={startStreaming}
                disabled={isStarted}
                sx={{
                  bgcolor: "#00ff88",
                  color: "#000",
                  fontWeight: 700,
                  px: 3,
                  py: 0.75,
                  fontSize: "0.8rem",
                  letterSpacing: 1,
                  fontFamily: '"Orbitron", monospace',
                  "&:hover": {
                    bgcolor: "#00dd77",
                  },
                  "&:disabled": {
                    bgcolor: "#004433",
                    color: "#006655",
                  },
                }}
              >
                START
              </Button>
              <Button
                variant="contained"
                onClick={stopStreaming}
                disabled={!isStarted}
                sx={{
                  bgcolor: "#ff3366",
                  color: "#fff",
                  fontWeight: 700,
                  px: 3,
                  py: 0.75,
                  fontSize: "0.8rem",
                  letterSpacing: 1,
                  fontFamily: '"Orbitron", monospace',
                  "&:hover": {
                    bgcolor: "#dd2255",
                  },
                  "&:disabled": {
                    bgcolor: "#442233",
                    color: "#663344",
                  },
                }}
              >
                STOP
              </Button>
            </Box>
          </Box>
        </Box>

        {/* Centered Content */}
        <Container
          maxWidth="lg"
          sx={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          {/* Canvas for frame capture (hidden) */}
          <canvas ref={captureCanvasRef} style={{ display: "none" }} />

          {/* Video Preview Modal - Shows during streaming */}
          <Box
            sx={{
              position: "fixed",
              bottom: 20,
              right: 20,
              width: 240,
              borderRadius: 2,
              overflow: "hidden",
              border: "2px solid #00ff88",
              boxShadow: "0 0 20px #00ff8860",
              bgcolor: "#000",
              zIndex: 1000,
              transition: "transform 0.3s ease, opacity 0.3s ease",
              display: showVideoPreview ? "block" : "none",
              "&:hover": {
                transform: showVideoPreview ? "scale(1.05)" : "none",
                boxShadow: showVideoPreview ? "0 0 30px #00ff88aa" : "none",
              },
            }}
          >
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{
                width: "100%",
                height: "auto",
                display: "block",
              }}
            />
            {showVideoPreview && (
              <Box
                sx={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  bgcolor: "rgba(0, 0, 0, 0.7)",
                  p: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    color: "#00ff88",
                    fontSize: "0.65rem",
                    fontWeight: 600,
                    letterSpacing: 1,
                  }}
                >
                  🔴 LIVE
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    color: "#8899aa",
                    fontSize: "0.6rem",
                  }}
                >
                  Frames: {apiStatus.framesSent}
                </Typography>
              </Box>
            )}
          </Box>

          {/* Heart Rate Warning Alert Modal - Top Left */}
          <AlertModal
            show={heartRateWarning !== null}
            type="warning"
            title="⚠️ HEART RATE WARNING ⚠️"
            message={
              heartRateWarning === "low" ? (
                <>
                  Your heart rate is below normal levels (
                  <span style={{ color: "#ff3366", fontWeight: 700 }}>
                    {healthMetrics.heartRate} BPM
                  </span>
                  ).
                  <br />
                  <br />
                  Consider stopping for a coffee break to boost your alertness
                  and energy levels.
                </>
              ) : heartRateWarning === "high" ? (
                <>
                  Your heart rate is elevated (
                  <span style={{ color: "#ff3366", fontWeight: 700 }}>
                    {healthMetrics.heartRate} BPM
                  </span>
                  ).
                  <br />
                  <br />
                  Please try to relax by taking deep breaths and consider
                  parking the vehicle nearby.
                </>
              ) : null
            }
            position={{ top: 20, left: 20 }}
            color="#ff3366"
          />

          {/* rPPG Waveform - No borders, no labels */}
          <Box sx={{ mb: 4, display: "flex", justifyContent: "center" }}>
            <canvas
              ref={canvasRef}
              style={{
                width: "100%",
                maxWidth: "1000px",
                height: "auto",
                background: "transparent",
                display: "block",
                filter: "drop-shadow(0 0 15px #00ff8860)",
              }}
              width={1000}
              height={200}
            />
          </Box>

          {/* Health Metrics - 3x2 grid with internal borders only */}
          <Box
            sx={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Box
              sx={{
                width: "100%",
                maxWidth: 1000,
                overflow: "hidden",
                bgcolor: "rgba(18, 22, 31, 0.4)",
              }}
            >
              <Grid container spacing={0} sx={{ width: "100%" }}>
                {[
                  // 0: Heart Rate
                  <MetricCard
                    icon={Favorite}
                    title="Heart Rate"
                    value={
                      healthMetrics.heartRate !== null
                        ? healthMetrics.heartRate
                        : apiStatus.isStreaming
                        ? "..."
                        : "--"
                    }
                    unit="BPM"
                    status={
                      healthMetrics.heartRate &&
                      healthMetrics.heartRate > 60 &&
                      healthMetrics.heartRate < 100
                        ? "normal"
                        : "warning"
                    }
                    color="#ff3366"
                    isRealData={healthMetrics.heartRate !== null}
                  />,
                  // 1: Blood Pressure (Dummy)
                  <MetricCard
                    icon={MonitorHeart}
                    title="Blood Pressure"
                    value={`${healthMetrics.bloodPressure.systolic}/${healthMetrics.bloodPressure.diastolic}`}
                    unit="mmHg"
                    status="normal"
                    color="#00ff88"
                    isRealData={false}
                  />,
                  // 2: Oxygen Level (Dummy)
                  <MetricCard
                    icon={Opacity}
                    title="Oxygen Level"
                    value={healthMetrics.oxygenLevel}
                    unit="%"
                    status={
                      healthMetrics.oxygenLevel >= 95 ? "normal" : "warning"
                    }
                    color="#00d9ff"
                    isRealData={false}
                  />,
                  // 3: Respiratory Rate (Dummy)
                  <MetricCard
                    icon={Air}
                    title="Respiratory Rate"
                    value={healthMetrics.respiratoryRate}
                    unit="breaths/min"
                    status="normal"
                    color="#ffaa00"
                    isRealData={false}
                  />,
                  // 4: Temperature (Dummy)
                  <MetricCard
                    icon={Thermostat}
                    title="Temperature"
                    value={healthMetrics.temperature}
                    unit="°F"
                    status="normal"
                    color="#ff66ff"
                    isRealData={false}
                  />,
                  // 5: Stress Level (custom card)
                  <Card
                    elevation={0}
                    sx={{
                      height: "100%",
                      bgcolor: "transparent",
                      border: "none",
                    }}
                  >
                    <CardContent
                      sx={{
                        p: 0,
                        "&:last-child": { pb: 0 },
                        height: "100%",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "center",
                      }}
                    >
                      <Box display="flex" alignItems="center" gap={1.5}>
                        <Box
                          sx={{
                            bgcolor: "#9966ff20",
                            borderRadius: 1.5,
                            p: 1,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            border: "2px solid #9966ff",
                          }}
                        >
                          <Psychology sx={{ color: "#9966ff", fontSize: 28 }} />
                        </Box>
                        <Box flex={1}>
                          <Typography
                            variant="caption"
                            sx={{
                              color: "#8899aa",
                              textTransform: "uppercase",
                              fontWeight: 600,
                              letterSpacing: 1,
                              fontSize: "0.65rem",
                            }}
                          >
                            Stress Level
                          </Typography>
                          <Box display="flex" alignItems="baseline" gap={0.5}>
                            <Typography
                              variant="h4"
                              fontWeight={700}
                              sx={{
                                lineHeight: 1,
                                color: "#9966ff",
                                fontFamily: '"Orbitron", monospace',
                                textShadow: "0 0 10px #9966ff80",
                              }}
                            >
                              {healthMetrics.stressLevel}
                            </Typography>
                          </Box>
                        </Box>
                        <Box
                          sx={{
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            bgcolor: "#ff3366",
                            boxShadow: "0 0 12px #ff3366",
                            animation: "pulse 2s infinite",
                          }}
                        />
                      </Box>
                    </CardContent>
                  </Card>,
                ].map((content, i) => {
                  // 3x2 grid (3 columns, 2 rows)
                  const borderColor = "#00ff8840";
                  const col = i % 3; // column index (0, 1, 2)
                  const row = Math.floor(i / 3); // row index (0, 1)

                  const hasRightBorder = col < 2; // right border for columns 0 and 1
                  const hasBottomBorder = row < 1; // bottom border for row 0

                  return (
                    <Grid
                      key={i}
                      item
                      xs={4}
                      sx={{
                        borderRight: hasRightBorder
                          ? `1px solid ${borderColor}`
                          : "none",
                        borderBottom: hasBottomBorder
                          ? `1px solid ${borderColor}`
                          : "none",
                        minHeight: 140,
                        maxWidth: "33.333333% !important",
                        flexBasis: "33.333333% !important",
                      }}
                    >
                      <Box sx={{ p: 3, height: "100%", width: "100%" }}>
                        {content}
                      </Box>
                    </Grid>
                  );
                })}
              </Grid>
            </Box>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
