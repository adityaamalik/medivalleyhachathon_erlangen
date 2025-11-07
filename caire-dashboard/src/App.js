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
    heartRate: 72,
    bloodPressure: { systolic: 120, diastolic: 80 },
    oxygenLevel: 98,
    respiratoryRate: 16,
    temperature: 98.6,
    stressLevel: "Low",
  });

  const [rppgData, setRppgData] = useState([]);
  const canvasRef = useRef(null);

  // Simulate real-time updates (you'll replace this with actual rPPG data later)
  useEffect(() => {
    const interval = setInterval(() => {
      setHealthMetrics((prev) => ({
        ...prev,
        heartRate: 70 + Math.floor(Math.random() * 10),
        oxygenLevel: 96 + Math.floor(Math.random() * 3),
      }));

      // Simulate rPPG data points
      setRppgData((prev) => {
        const newData = [
          ...prev,
          Math.sin(Date.now() / 200) * 50 + Math.random() * 10,
        ];
        return newData.slice(-100); // Keep last 100 points
      });
    }, 50);

    return () => clearInterval(interval);
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
      // Draw glowing background line
      ctx.strokeStyle = "#00ff8840";
      ctx.lineWidth = 8;
      ctx.beginPath();

      const step = width / rppgData.length;
      rppgData.forEach((value, index) => {
        const x = index * step;
        const y = height / 2 + value;

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
        const y = height / 2 + value;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }
  }, [rppgData]);

  const MetricCard = ({ icon: Icon, title, value, unit, status, color }) => (
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
              bgcolor: status === "normal" ? "#00ff88" : "#ffaa00",
              boxShadow:
                status === "normal" ? "0 0 12px #00ff88" : "0 0 12px #ffaa00",
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
            CAIRE DRIVER HEALTH MONITOR
          </Typography>
          <Typography
            variant="body2"
            sx={{ color: "#00d9ff", letterSpacing: 2, mt: 0.5 }}
          >
            REAL-TIME BIOMETRIC ANALYSIS SYSTEM
          </Typography>
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
                    value={healthMetrics.heartRate}
                    unit="BPM"
                    status={
                      healthMetrics.heartRate > 60 &&
                      healthMetrics.heartRate < 100
                        ? "normal"
                        : "warning"
                    }
                    color="#ff3366"
                  />,
                  // 1: Blood Pressure
                  <MetricCard
                    icon={MonitorHeart}
                    title="Blood Pressure"
                    value={`${healthMetrics.bloodPressure.systolic}/${healthMetrics.bloodPressure.diastolic}`}
                    unit="mmHg"
                    status="normal"
                    color="#00ff88"
                  />,
                  // 2: Oxygen Level
                  <MetricCard
                    icon={Opacity}
                    title="Oxygen Level"
                    value={healthMetrics.oxygenLevel}
                    unit="%"
                    status={
                      healthMetrics.oxygenLevel >= 95 ? "normal" : "warning"
                    }
                    color="#00d9ff"
                  />,
                  // 3: Respiratory Rate
                  <MetricCard
                    icon={Air}
                    title="Respiratory Rate"
                    value={healthMetrics.respiratoryRate}
                    unit="breaths/min"
                    status="normal"
                    color="#ffaa00"
                  />,
                  // 4: Temperature
                  <MetricCard
                    icon={Thermostat}
                    title="Temperature"
                    value={healthMetrics.temperature}
                    unit="°F"
                    status="normal"
                    color="#ff66ff"
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
                            bgcolor: "#00ff88",
                            boxShadow: "0 0 12px #00ff88",
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
                      <Box sx={{ p: 3, height: "100%", width: "100%" }}>{content}</Box>
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
