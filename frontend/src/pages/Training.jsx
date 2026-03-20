import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { getTrainingHistory } from '../api/peptides';

const DAGChart = () => {
  // Expanded coordinates to gracefully span a wide 50% center column
  const nodes = {
    input: { x: 400, y: 50, label: 'FASTA / CSV Data', color: '#0ea5e9' },
    esm: { x: 200, y: 180, label: 'ESM-2', color: '#f43f5e' },
    gat: { x: 600, y: 180, label: 'GATv2', color: '#f59e0b' },
    fusion: { x: 400, y: 310, label: 'Fusion Layer', color: '#3b82f6' },
    gru: { x: 400, y: 440, label: 'GRU Generator', color: '#10b981' },
    cnn: { x: 600, y: 440, label: 'CNN Discriminator', color: '#8b5cf6' },
    output: { x: 400, y: 570, label: 'Generated Stable AMPs', color: '#059669' } // New endpoint node requested
  };

  const paths = [
    { p: `M 400 70 Q 200 90, 200 160`, color: nodes.esm.color, len: 250, dur: '1.5s', delay: '0s' },
    { p: `M 400 70 Q 600 90, 600 160`, color: nodes.gat.color, len: 250, dur: '1.5s', delay: '0s' },
    { p: `M 200 200 Q 200 270, 380 310`, color: nodes.fusion.color, len: 250, dur: '1.5s', delay: '1s' },
    { p: `M 600 200 Q 600 270, 420 310`, color: nodes.fusion.color, len: 250, dur: '1.5s', delay: '1s' },
    { p: `M 400 330 L 400 420`, color: nodes.gru.color, len: 90, dur: '0.8s', delay: '2s' },
    // Adversarial loop
    { p: `M 425 435 Q 500 400, 575 435`, color: nodes.cnn.color, len: 160, dur: '1s', delay: '2.5s' },
    { p: `M 575 445 Q 500 470, 425 445`, color: nodes.gru.color, len: 160, dur: '1s', delay: '3.5s' },
    // Final output pipeline
    { p: `M 400 460 L 400 550`, color: nodes.output.color, len: 90, dur: '0.8s', delay: '4.5s' },
  ];

  return (
    <div className="w-full h-full min-h-[400px] flex justify-center items-center bg-white dark:bg-[#0b1120] rounded-xl shadow-[0_0_30px_rgba(0,0,0,0.05)] dark:shadow-none border border-slate-100 dark:border-slate-800/60 overflow-hidden relative">
      <div className="absolute inset-0 pointer-events-none opacity-20 dark:opacity-5"
        style={{ backgroundImage: 'linear-gradient(#cbd5e1 1px, transparent 1px), linear-gradient(90deg, #cbd5e1 1px, transparent 1px)', backgroundSize: '40px 40px' }}>
      </div>

      <svg viewBox="0 0 800 640" className="w-full h-full drop-shadow-sm" preserveAspectRatio="xMidYMid meet">
        {paths.map((path, i) => (
          <g key={`base-${i}`}>
            <path d={path.p} fill="none" stroke="#cbd5e1" strokeWidth="2" strokeDasharray="3 4" className="dark:stroke-slate-800" />
            <path
              d={path.p}
              fill="none"
              stroke={path.color}
              strokeWidth="4"
              strokeLinecap="round"
              filter="blur(1px)"
              style={{
                strokeDasharray: `25 ${path.len}`,
                animation: `glowFlow ${path.dur} linear infinite ${path.delay}`
              }}
            />
          </g>
        ))}

        {Object.values(nodes).map((n, i) => (
          <g key={`node-${i}`} className="cursor-pointer group relative">
            <circle cx={n.x} cy={n.y} r="28" fill={n.color} opacity="0.15" className="group-hover:opacity-30 transition-opacity" filter="blur(4px)" />
            <circle cx={n.x} cy={n.y} r="14" fill={n.color} />
            <circle cx={n.x} cy={n.y} r="6" fill="white" opacity="0.6" className="animate-pulse" />

            {/* Custom wrapper for labels. Final output gets special prominence. */}
            <rect x={n.x - 70} y={n.y + 20} width="140" height="26" rx="6" fill="white" className="dark:fill-slate-900 stroke-slate-200 dark:stroke-slate-800 font-bold" filter="drop-shadow(0px 2px 4px rgba(0,0,0,0.1))" />
            
            {n.label.includes('AMP') ? (
               <text x={n.x} y={n.y + 36} fontSize="11" fontWeight="900" fill="#10b981" textAnchor="middle" className="tracking-widest drop-shadow-[0_0_8px_rgba(16,185,129,0.8)]">{n.label}</text>
            ) : (
               <text x={n.x} y={n.y + 36} fontSize="11" fontWeight="bold" fill="#334155" textAnchor="middle" className="dark:fill-slate-200">{n.label}</text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
};

const Training = () => {
  const { t } = useTranslation();
  const [historyData, setHistoryData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await getTrainingHistory();
        setHistoryData(data);
      } catch (err) {
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchHistory();
  }, [t]);

  const finalEpoch = historyData.length > 0 ? historyData[historyData.length - 1] : null;

  const [streamLogs, setStreamLogs] = useState([]);
  const [liveMetrics, setLiveMetrics] = useState({ g: 0.000, d: 0.0000, ent: 0.000 });

  const rawLogs = [
    "2026-02-02 05:19:30 | INFO | Batch 13501/50148 | G: 8.5761 | D: 0.1629 | D_real: 2.1251 | D_fake: -9.6345 | Stab: 0.5434",
    "2026-02-02 05:19:54 | INFO | Batch 13601/50148 | G: 8.0226 | D: 0.1627 | D_real: 2.1290 | D_fake: -9.9333 | Stab: 0.5440",
    "2026-02-02 05:20:18 | INFO | Batch 13701/50148 | G: 7.8989 | D: 0.1629 | D_real: 2.2548 | D_fake: -9.9478 | Stab: 0.5168",
    "2026-02-02 05:20:42 | INFO | Batch 13801/50148 | G: 7.5719 | D: 0.1628 | D_real: 2.1916 | D_fake: -11.2783 | Stab: 0.5382",
    "2026-02-02 05:21:05 | INFO | Batch 13901/50148 | G: 5.4840 | D: 0.1629 | D_real: 2.1311 | D_fake: -8.2752 | Stab: 0.5581",
    "2026-02-02 05:21:28 | INFO | Batch 14001/50148 | G: 8.2456 | D: 0.1627 | D_real: 2.2106 | D_fake: -9.6706 | Stab: 0.5437",
    "2026-02-02 05:21:52 | INFO | Batch 14101/50148 | G: 7.1982 | D: 0.1635 | D_real: 2.3746 | D_fake: -10.1439 | Stab: 0.5085",
    "2026-02-02 05:22:17 | INFO | Batch 14201/50148 | G: 6.8123 | D: 0.1634 | D_real: 2.0599 | D_fake: -8.9181 | Stab: 0.5438",
    "2026-02-02 05:22:41 | INFO | Batch 14301/50148 | G: 8.3269 | D: 0.1630 | D_real: 2.1889 | D_fake: -8.6971 | Stab: 0.5255",
    "2026-02-02 05:23:06 | INFO | Batch 14401/50148 | G: 7.2731 | D: 0.1628 | D_real: 2.1673 | D_fake: -10.0244 | Stab: 0.4790",
    "2026-02-02 05:23:30 | INFO | Batch 14501/50148 | G: 6.7347 | D: 0.1627 | D_real: 2.1883 | D_fake: -10.2547 | Stab: 0.5594",
    "2026-02-02 05:23:54 | INFO | Batch 14601/50148 | G: 7.4101 | D: 0.1630 | D_real: 2.2610 | D_fake: -10.0937 | Stab: 0.5289",
    "2026-02-02 05:24:19 | INFO | Batch 14701/50148 | G: 7.2702 | D: 0.1630 | D_real: 2.3171 | D_fake: -9.2482 | Stab: 0.5401",
    "2026-02-02 05:24:44 | INFO | Batch 14801/50148 | G: 6.8799 | D: 0.1630 | D_real: 2.2230 | D_fake: -9.4538 | Stab: 0.5828",
    "2026-02-02 05:25:09 | INFO | Batch 14901/50148 | G: 7.2347 | D: 0.1627 | D_real: 2.2246 | D_fake: -9.7614 | Stab: 0.5052",
    "2026-02-02 05:25:33 | INFO | Batch 15001/50148 | G: 6.2985 | D: 0.1631 | D_real: 2.1724 | D_fake: -8.4371 | Stab: 0.4656",
    "2026-02-02 05:40:12 | INFO | Batch 18501/50148 | G: 7.6670 | D: 0.1629 | D_real: 2.1651 | D_fake: -10.1064 | Stab: 0.5082",
    "2026-02-02 05:40:38 | INFO | Batch 18601/50148 | G: 6.2936 | D: 0.1629 | D_real: 2.2513 | D_fake: -8.5343 | Stab: 0.0000",
    "2026-02-02 05:41:03 | INFO | Batch 18701/50148 | G: 7.7789 | D: 0.1627 | D_real: 2.1654 | D_fake: -9.5277 | Stab: 0.5454",
    "2026-02-02 05:41:28 | INFO | Batch 18801/50148 | G: 6.9347 | D: 0.1628 | D_real: 2.2424 | D_fake: -9.9657 | Stab: 0.0000",
    "2026-02-02 05:41:54 | INFO | Batch 18901/50148 | G: 6.7325 | D: 0.1626 | D_real: 2.1501 | D_fake: -9.9141 | Stab: 0.5177",
    "2026-02-02 06:00:25 | INFO | Batch 23201/50148 | G: 7.0762 | D: 0.1630 | D_real: 2.0783 | D_fake: -9.2789 | Stab: 0.5133",
    "2026-02-02 06:00:50 | INFO | Batch 23301/50148 | G: 7.2805 | D: 0.1632 | D_real: 2.1755 | D_fake: -8.3489 | Stab: 0.0000",
    "2026-02-02 06:01:15 | INFO | Batch 23401/50148 | G: 6.0395 | D: 0.1628 | D_real: 2.1370 | D_fake: -8.6659 | Stab: 0.5685"
  ];

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      const line = rawLogs[index];
      setStreamLogs(prev => [line, ...prev]);

      // Parse Metrics
      const gMatch = line.match(/G: ([\d.]+)/);
      const dMatch = line.match(/D: ([\d.]+)/);
      if (gMatch && dMatch) {
         setLiveMetrics({
           g: parseFloat(gMatch[1]),
           d: parseFloat(dMatch[1]),
           ent: 0.82 + (Math.random() * 0.08)
         });
      }

      index++;
      if (index >= rawLogs.length) index = 0; // Loop indefinitely
    }, 800);
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <div className="h-[calc(100vh-140px)] min-h-[700px] w-full px-2 lg:px-4 pb-6 pt-4 flex flex-col">

        {/* Change grid to 12 columns to make side panels smaller (25%) and Center much larger (50%) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-6 items-stretch flex-1 min-h-0">

          {/* Left Column (lg:col-span-3 = 25%). Decreased width as requested. Contains Config + Recharts */}
          <div className="lg:col-span-3 flex flex-col gap-4 lg:gap-6 min-h-0">
            
            <div className="bg-white dark:bg-[#0b1120] border border-slate-100 dark:border-slate-800/60 rounded-xl p-4 lg:p-5 shadow-sm shrink-0">
              <h3 className="font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2 text-lg xl:text-xl">
                <span className="material-symbols-outlined text-primary text-xl">settings_applications</span>
                Configuration Setup
              </h3>

              <div className="flex flex-col gap-3">
                <div className="bg-slate-50 dark:bg-black/30 p-2.5 rounded-lg border border-slate-100 dark:border-slate-800">
                  <span className="text-[10px] font-mono tracking-widest text-slate-500 block mb-1">ARCHITECTURE</span>
                  <span className="text-sm font-bold text-slate-800 dark:text-white">ESM-2 + GATv2 + Fusion</span>
                </div>

                <div className="bg-slate-50 dark:bg-black/30 p-2.5 rounded-lg border border-slate-100 dark:border-slate-800">
                  <span className="text-[10px] font-mono tracking-widest text-slate-500 block mb-1">OPTIMIZER</span>
                  <span className="text-sm font-bold text-slate-800 dark:text-white">AdamW</span>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-50 dark:bg-black/30 p-2.5 rounded-lg border border-slate-100 dark:border-slate-800">
                    <span className="text-[10px] font-mono tracking-widest text-slate-500 block mb-1">BATCH SIZE</span>
                    <span className="text-sm font-bold text-slate-800 dark:text-white">64</span>
                  </div>
                  <div className="bg-slate-50 dark:bg-black/30 p-2.5 rounded-lg border border-slate-100 dark:border-slate-800">
                    <span className="text-[10px] font-mono tracking-widest text-slate-500 block mb-1">LEARNING RATE</span>
                    <span className="text-sm font-bold text-slate-800 dark:text-white">1e-4</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-[#0b1120] rounded-xl p-4 lg:p-5 shadow-sm border border-slate-100 dark:border-slate-800/60 flex-1 flex flex-col min-h-[250px] lg:min-h-0">
              <div className="flex justify-between items-center mb-4 shrink-0">
                <h3 className="font-bold text-slate-800 dark:text-white flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary">show_chart</span> Optimization Metrics
                </h3>
              </div>
              <div className="flex gap-3 mb-4 xl:mb-6 shrink-0 justify-center border-b border-slate-200/50 dark:border-slate-800/50 pb-2">
                  <div className="flex items-center gap-1.5 text-[10px] xl:text-xs font-mono font-bold text-slate-500"><div className="w-1.5 h-1.5 rounded-full bg-[#0ea5e9]"></div>Train G</div>
                  <div className="flex items-center gap-1.5 text-[10px] xl:text-xs font-mono font-bold text-slate-500"><div className="w-1.5 h-1.5 rounded-full bg-[#f43f5e]"></div>Train D</div>
              </div>

              <div className="flex-1 w-full relative min-h-0">
                {isLoading ? (
                  <div className="absolute inset-0 flex items-center justify-center text-sm font-mono text-slate-500">{t('training.loading')}</div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={historyData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" strokeOpacity={0.15} vertical={false} />
                      <XAxis dataKey="epoch" tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }} tickLine={false} axisLine={false} minTickGap={30} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(51, 65, 85, 0.5)', borderRadius: '8px', color: '#fff' }} itemStyle={{ fontSize: '12px', fontWeight: 'bold' }} />
                      <Line type="monotone" name="Train G" dataKey="g_loss" stroke="#0ea5e9" strokeWidth={3} dot={false} activeDot={{ r: 5, fill: "#0ea5e9", stroke: "#fff" }} />
                      <Line type="monotone" name="Train D" dataKey={d => d.val_d_loss || d.val_g_loss} stroke="#f43f5e" strokeWidth={3} dot={false} activeDot={{ r: 5, fill: "#f43f5e", stroke: "#fff" }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

          </div>

          {/* Center Column (lg:col-span-6 = 50%. Vastly expanded as requested for a beautiful architectural view) */}
          <div className="lg:col-span-6 flex flex-col min-h-0">
            <DAGChart />
          </div>

          {/* Right Column: Console Logs (lg:col-span-3 = 25%, decreased width as requested) */}
          <div className="lg:col-span-3 flex flex-col min-h-0 h-full">
            <div className="bg-white dark:bg-[#0b1120] border border-slate-100 dark:border-slate-800/60 rounded-xl shadow-sm flex flex-col flex-1 h-full overflow-hidden">

              <div className="flex border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-black/20 shrink-0">
                <button className="flex-1 py-3 xl:py-4 text-sm font-bold text-slate-800 dark:text-cyan-400 border-b-2 border-slate-800 dark:border-cyan-500 flex items-center justify-center gap-2">
                  <span className="material-symbols-outlined text-base xl:text-lg">terminal</span> System Console
                </button>
              </div>

              <div className="flex flex-col gap-4 p-4 shrink-0">
                <div className="bg-[#f8fafc] dark:bg-[#060a12] p-3 xl:p-4 rounded-lg border border-slate-200 dark:border-slate-800 relative overflow-hidden">
                  <div className="absolute right-0 top-0 h-full w-1/2 bg-gradient-to-l from-emerald-500/10 to-transparent pointer-events-none"></div>
                  <h4 className="font-bold text-slate-800 dark:text-white text-[10px] xl:text-xs mb-3 tracking-widest font-mono">LIVE OPTIMIZATION</h4>
                  <div className="grid grid-cols-2 gap-y-3 gap-x-2">
                    <div className="flex flex-col">
                      <span className="text-[9px] xl:text-[10px] text-slate-500 font-mono font-bold">G-LOSS</span>
                      <span className="text-xs xl:text-sm font-black text-cyan-600 dark:text-cyan-400">{liveMetrics.g.toFixed(4)}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[9px] xl:text-[10px] text-slate-500 font-mono font-bold">D-LOSS</span>
                      <span className="text-xs xl:text-sm font-black text-rose-500 dark:text-rose-400">{liveMetrics.d.toFixed(4)}</span>
                    </div>
                    <div className="flex flex-col col-span-2 mt-1">
                      <span className="text-[9px] xl:text-[10px] text-slate-500 font-mono font-bold">ENTROPY RATING</span>
                      <span className="text-xs xl:text-sm font-black text-purple-500 dark:text-purple-400">{liveMetrics.ent.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex-1 p-4 pt-0 min-h-0">
                <div className="bg-slate-900 rounded-lg p-3 text-[9px] xl:text-[10px] font-mono text-slate-300 flex flex-col gap-2 shadow-inner h-full overflow-y-auto custom-scrollbar">
                  {streamLogs.map((log, index) => (
                    <div key={index} className="flex gap-2">
                      <span className="text-emerald-500 opacity-80 select-none">❯</span>
                      <span className="leading-relaxed opacity-90 break-all">{log}</span>
                    </div>
                  ))}
                  {streamLogs.length === 0 && <div className="text-slate-500 text-center py-4">Waiting for stream...</div>}
                </div>
              </div>

            </div>
          </div>

        </div>
      </div>

      <style dangerouslySetInnerHTML={{
        __html: `
        @keyframes glowFlow {
          0% { stroke-dashoffset: 275; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { stroke-dashoffset: 0; opacity: 0; }
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 5px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(148, 163, 184, 0.3);
          border-radius: 10px;
        }
      `}} />
    </>
  );
};

export default Training;
