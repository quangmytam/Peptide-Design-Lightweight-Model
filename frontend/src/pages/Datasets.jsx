import React from 'react';
import { useTranslation } from 'react-i18next';

const FuturisticPanel = ({ title, children, className }) => (
  // Added flex-col and h-full so the panel expands taking remaining space
  <div className={`relative bg-white/60 dark:bg-[#060a12]/80 backdrop-blur-md border border-slate-200 dark:border-cyan-900/40 p-4 xl:p-6 shadow-sm dark:shadow-[0_0_20px_rgba(8,145,178,0.1)] rounded-sm overflow-hidden group flex flex-col h-full ${className}`}>
    <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-primary/50 dark:border-cyan-500/70"></div>
    <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-primary/50 dark:border-cyan-500/70"></div>
    <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-primary/50 dark:border-cyan-500/70"></div>
    <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-primary/50 dark:border-cyan-500/70"></div>

    <div className="mb-4 pb-2 border-b border-slate-200 dark:border-slate-800/50 flex items-center justify-between shrink-0">
      <h3 className="font-mono text-sm md:text-base font-bold tracking-[0.2em] text-slate-800 dark:text-cyan-400 uppercase">{title}</h3>
      <div className="w-8 h-[2px] bg-slate-300 dark:bg-cyan-900/80"></div>
    </div>
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {children}
    </div>
  </div>
);

const AnimatedPeptide = () => {
  return (
    <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 md:w-96 md:h-96 bg-cyan-500/5 dark:bg-cyan-500/10 blur-[100px] rounded-full pointer-events-none"></div>

      {/* 3D Peptide Hologram scale responsive */}
      <div className="relative animate-[spin_20s_linear_infinite] w-[260px] h-[260px] md:w-[300px] md:h-[300px] lg:w-[350px] lg:h-[350px] perspective-1000 z-10">
        <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible drop-shadow-[0_0_8px_rgba(34,211,238,0.4)] dark:drop-shadow-[0_0_12px_rgba(34,211,238,0.7)]">
          <defs>
            <linearGradient id="stickGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#818cf8" stopOpacity="0.5" />
            </linearGradient>
          </defs>

          <path d="M 10 50 Q 25 15, 50 35 T 90 50" fill="none" stroke="url(#stickGrad)" strokeWidth="1.5" className="animate-[pulse_3s_infinite]" />
          <path d="M 50 35 Q 75 75, 85 90" fill="none" stroke="url(#stickGrad)" strokeWidth="1" className="opacity-70" />
          <path d="M 90 50 Q 95 20, 75 10" fill="none" stroke="url(#stickGrad)" strokeWidth="1" className="opacity-50" />

          <circle cx="10" cy="50" r="3.5" fill="#22d3ee" className="animate-ping shadow-lg" style={{ animationDuration: '3s' }} />
          <circle cx="10" cy="50" r="3" fill="#fff" />
          <circle cx="50" cy="35" r="4.5" fill="#38bdf8" />
          <circle cx="90" cy="50" r="3" fill="#818cf8" />
          <circle cx="85" cy="90" r="2.5" fill="#c084fc" className="animate-[pulse_2s_infinite]" />

          <circle cx="30" cy="27" r="1.5" fill="#e879f9" opacity="0.9" />
          <line x1="25" y1="35" x2="30" y2="27" stroke="url(#stickGrad)" strokeWidth="0.5" strokeDasharray="1 1" />

          <circle cx="70" cy="42" r="2" fill="#a78bfa" opacity="0.9" />
          <line x1="60" y1="38" x2="70" y2="42" stroke="url(#stickGrad)" strokeWidth="0.5" strokeDasharray="1 1" />

          <circle cx="50" cy="35" r="1" fill="#fff" className="animate-[ping_2s_infinite]" />
        </svg>
      </div>

      <div className="absolute top-4 left-[2%] text-[10px] md:text-xs font-mono text-slate-500 dark:text-cyan-500/60 tracking-[0.3em] font-bold hidden md:block">
        T: 34.1 C <br /> <br /> PEPTIDE <br /> SHORT_SEQ
      </div>
      <div className="absolute bottom-4 right-[2%] text-[10px] md:text-xs font-mono text-slate-500 dark:text-emerald-500/60 tracking-[0.3em] text-right font-bold hidden md:block">
        E.STABLE <br /> α-HELIX <br /> METRICS OK
      </div>

      {/* Sci-fi Target Reticle */}
      <div className="absolute w-[90%] h-[90%] max-w-[400px] max-h-[400px] border-[0.5px] border-slate-300 dark:border-cyan-800/30 rounded-full flex items-center justify-center pointer-events-none mt-2">
        <div className="w-full h-[0.5px] bg-slate-200 dark:bg-cyan-800/20 absolute"></div>
        <div className="h-full w-[0.5px] bg-slate-200 dark:bg-cyan-800/20 absolute"></div>
        <div className="absolute top-0 w-3 h-px bg-slate-500 dark:bg-cyan-500"></div>
        <div className="absolute bottom-0 w-3 h-px bg-slate-500 dark:bg-cyan-500"></div>
        <div className="absolute left-0 h-3 w-px bg-slate-500 dark:bg-cyan-500"></div>
        <div className="absolute right-0 h-3 w-px bg-slate-500 dark:bg-cyan-500"></div>
      </div>
    </div>
  );
};

const Datasets = () => {
  const { t } = useTranslation();

  return (
    // Height matched exactly to 100vh - header/footer height (~150px) to prevent scroll
    <div className="h-[calc(100vh-140px)] min-h-[600px] lg:min-h-[700px] py-4 lg:py-6 px-2 lg:px-4 bg-slate-50 dark:bg-[#030712] relative overflow-hidden transition-colors duration-500 flex flex-col">

      <div className="absolute inset-0 pointer-events-none opacity-[0.03] dark:opacity-[0.05]"
        style={{ backgroundImage: 'linear-gradient(#000 1px, transparent 1px), linear-gradient(90deg, #000 1px, transparent 1px)', backgroundSize: '40px 40px' }}>
      </div>

      <div className="w-full flex-1 flex flex-col relative z-10 min-h-0">
        <div className="grid grid-cols-1 lg:grid-cols-4 xl:grid-cols-12 gap-4 lg:gap-6 items-stretch h-full flex-1 min-h-0">

          {/* Left Panel: FOSSIL DATA */}
          <div className="lg:col-span-1 xl:col-span-3 flex flex-col h-full min-h-0">
            <FuturisticPanel title="Bio-PDB Analytics">

              <div className="flex flex-col gap-4 mb-6 shrink-0 mt-2">
                <div className="flex flex-col gap-1 border-b border-slate-200 dark:border-slate-800 pb-3">
                  <span className="text-[10px] xl:text-xs font-mono tracking-[0.1em] text-slate-500 dark:text-slate-400 uppercase font-bold">Total Samples</span>
                  <span className="text-2xl md:text-3xl xl:text-4xl font-black text-slate-800 dark:text-white leading-none tracking-tight">184,460</span>
                </div>

                <div className="flex justify-between items-end border-b border-slate-200 dark:border-slate-800 pb-3">
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] xl:text-xs font-mono tracking-[0.1em] text-emerald-600 dark:text-emerald-500 font-bold uppercase">AMP (50%)</span>
                    <span className="text-xl md:text-2xl xl:text-3xl font-black text-slate-800 dark:text-emerald-50 leading-none">92,230</span>
                  </div>
                  <span className="text-[9px] md:text-[10px] xl:text-xs font-mono tracking-widest font-bold text-emerald-600 bg-emerald-100 dark:bg-emerald-900/30 px-2 py-0.5 border border-emerald-200 dark:border-emerald-800/50 rounded-sm">BALANCED</span>
                </div>

                <div className="flex justify-between items-end border-b border-slate-200 dark:border-slate-800 pb-3">
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] xl:text-xs font-mono tracking-[0.1em] text-rose-500 font-bold uppercase">Non-AMP (50%)</span>
                    <span className="text-xl md:text-2xl xl:text-3xl font-black text-slate-800 dark:text-rose-50 leading-none">92,230</span>
                  </div>
                  <span className="text-[9px] md:text-[10px] xl:text-xs font-mono tracking-widest font-bold text-rose-500 bg-rose-100 dark:bg-rose-900/30 px-2 py-0.5 border border-rose-200 dark:border-rose-800/50 rounded-sm">BALANCED</span>
                </div>
              </div>

              <div className="w-full flex-1 bg-slate-100 dark:bg-[#02040a] border border-slate-200 dark:border-slate-800/70 flex items-center justify-center p-3 relative group overflow-hidden rounded-sm shadow-inner min-h-0">
                <div className="absolute top-2 left-2 text-[8px] md:text-[10px] xl:text-xs font-mono text-slate-500 dark:text-slate-400 z-10 tracking-widest font-bold">AMINO ACID DIST.</div>
                <img
                  alt="Amino Acid Distribution"
                  className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-700"
                  src={`${import.meta.env.BASE_URL}aa_distribution.png`}
                  onError={(e) => { e.target.style.display = 'none'; e.target.parentElement.innerHTML += '<div class="absolute inset-0 flex items-center justify-center text-slate-400 font-mono text-xs">Waiting for Scan...</div>' }}
                />
              </div>
            </FuturisticPanel>
          </div>

          {/* Center Panel: Hologram Visual */}
          <div className="lg:col-span-2 xl:col-span-6 flex flex-col items-center justify-center relative h-full py-2">

            {/* === SWAPPED POSITIONS AS REQUESTED === */}
            {/* TOXICITY GUARD was Bottom-Right, now Top-Left (shifted right 15px) */}
            <div className="absolute top-0 left-2 lg:left-6 translate-x-[500px] z-20 flex flex-col items-end text-right bg-slate-50/80 dark:bg-[#030712]/60 p-2 xl:p-3 rounded backdrop-blur-md border border-slate-200/80 dark:border-rose-900/40">
              <div className="text-[10px] xl:text-xs bg-rose-100 dark:bg-rose-900/30 text-rose-600 dark:text-rose-400 px-2 py-0.5 xl:py-1 border border-rose-300 dark:border-rose-800/50 tracking-[0.2em] mb-1.5 xl:mb-2 font-mono font-bold uppercase rounded-sm w-fit">
                TOXICITY GUARD
              </div>
              <div className="text-xs xl:text-sm text-slate-700 dark:text-slate-300 max-w-[160px] xl:max-w-[180px] leading-relaxed">Hemolytic score strongly maintained below the critical safety threshold.</div>
            </div>

            {/* PREFERRED SEQUENCE was Top-Left, now Bottom-Right (shifted left 15px) */}
            <div className="absolute bottom-4 right-2 lg:right-6 -translate-x-[500px] z-20 flex flex-col items-start text-left bg-slate-50/80 dark:bg-[#030712]/60 p-2 xl:p-3 rounded backdrop-blur-md border border-slate-200/80 dark:border-cyan-900/40">
              <div className="text-[10px] xl:text-xs font-mono text-cyan-600 dark:text-cyan-400 tracking-wider mb-1.5 xl:mb-2 flex items-center gap-2 font-bold uppercase">
                <span className="w-1.5 h-1.5 xl:w-2 xl:h-2 rounded-full bg-cyan-500 shadow-[0_0_10px_#06b6d4] animate-pulse"></span> PREFERRED SEQUENCE
              </div>
              <div className="text-xs xl:text-sm text-slate-700 dark:text-slate-300 max-w-[160px] xl:max-w-[180px] leading-relaxed">L, A, G, K residues globally dominant in stable configurations.</div>
            </div>

            <AnimatedPeptide />

          </div>

          {/* Right Panel: Feature Diagnostics */}
          <div className="lg:col-span-1 xl:col-span-3 flex flex-col h-full min-h-0">

            <FuturisticPanel title="Feature Diagnostics">
              <div className="flex flex-col gap-2 xl:gap-3 flex-1 mt-2 min-h-0">
                <div className="flex-1 bg-slate-100 dark:bg-[#02040a] border border-slate-200 dark:border-slate-800/70 p-2 xl:p-3 flex items-center justify-center relative rounded-sm shadow-inner group overflow-hidden min-h-0">
                  <span className="absolute top-1.5 left-1.5 text-[8px] xl:text-[9px] font-mono text-slate-500 dark:text-slate-400/80 uppercase tracking-widest font-bold z-10">Sequence Length</span>
                  <img
                    alt="Length"
                    className="w-full h-full object-contain p-2 mt-4"
                    src={`${import.meta.env.BASE_URL}length_hist.png`}
                    onError={(e) => { e.target.style.display = 'none'; e.target.parentElement.innerHTML += '<div class="absolute inset-0 flex items-center justify-center text-slate-400 font-mono text-[8px]">Scan Wait</div>' }}
                  />
                </div>

                <div className="flex-1 bg-slate-100 dark:bg-[#02040a] border border-slate-200 dark:border-slate-800/70 p-2 xl:p-3 flex items-center justify-center relative rounded-sm shadow-inner group overflow-hidden min-h-0">
                  <span className="absolute top-1.5 left-1.5 text-[8px] xl:text-[9px] font-mono text-slate-500 dark:text-slate-400/80 uppercase tracking-widest font-bold z-10">Instability Index</span>
                  <img
                    alt="Instability"
                    className="w-full h-full object-contain p-2 mt-4"
                    src={`${import.meta.env.BASE_URL}instability_hist.png`}
                    onError={(e) => { e.target.style.display = 'none'; e.target.parentElement.innerHTML += '<div class="absolute inset-0 flex items-center justify-center text-slate-400 font-mono text-[8px]">Scan Wait</div>' }}
                  />
                </div>

                <div className="flex-1 bg-slate-100 dark:bg-[#02040a] border border-slate-200 dark:border-slate-800/70 p-2 xl:p-3 flex items-center justify-center relative rounded-sm shadow-inner group overflow-hidden min-h-0">
                  <span className="absolute top-2 left-2 text-[9px] xl:text-[10px] font-mono text-slate-500 dark:text-slate-400/80 uppercase tracking-widest font-bold z-10">GRAVY Score Analysis</span>
                  <img
                    alt="GRAVY"
                    className="w-full h-full object-contain p-2 mt-4"
                    src={`${import.meta.env.BASE_URL}gravy_hist.png`}
                    onError={(e) => { e.target.style.display = 'none'; e.target.parentElement.innerHTML += '<div class="absolute inset-0 flex items-center justify-center text-slate-400 font-mono text-[10px]">Scanning Component...</div>' }}
                  />
                </div>
              </div>
            </FuturisticPanel>
          </div>

        </div>
      </div>
    </div>
  );
};

export default Datasets;
