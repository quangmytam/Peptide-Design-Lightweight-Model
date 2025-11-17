import React, { useState, useEffect, useRef } from 'react';
import Header from '../components/Header';
import Card from '../components/Card';
import PageTitle from '../components/PageTitle';
import SelectInput from '../components/SelectInput';
import TextInput from '../components/TextInput';
import Button from '../components/Button';
import MetricDisplay from '../components/MetricDisplay';

const Training = () => {
  const [epochs, setEpochs] = useState(100);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Idle');
  const [logs, setLogs] = useState([]);
  const trainingInterval = useRef(null);

  const startTraining = () => {
    setStatus('Training...');
    setLogs(['[INFO] Initializing LightGNN-Peptide model...', '[INFO] Loading dataset: BioPDB v1.2.', '[INFO] Found 15,789 peptide structures.']);
    trainingInterval.current = setInterval(() => {
      setCurrentEpoch((prev) => {
        if (prev < epochs) {
          const newEpoch = prev + 1;
          setProgress((newEpoch / epochs) * 100);
          setLogs((prevLogs) => [...prevLogs, `[TRAIN] Epoch ${newEpoch}/${epochs} - Loss: ${Math.random().toFixed(3)}, Val Acc: ${(Math.random() * 10 + 80).toFixed(2)}%`]);
          return newEpoch;
        }
        stopTraining('Completed');
        return prev;
      });
    }, 1000);
  };

  const pauseTraining = () => {
    if (status === 'Training...') {
      setStatus('Paused');
      clearInterval(trainingInterval.current);
    } else if (status === 'Paused') {
      setStatus('Training...'); // Resume
      startTraining();
    }
  };

  const stopTraining = (finalStatus = 'Stopped') => {
    setStatus(finalStatus);
    clearInterval(trainingInterval.current);
    if (finalStatus !== 'Completed') {
        setCurrentEpoch(0);
        setProgress(0);
    }
  };

  const resetTraining = () => {
    stopTraining('Idle');
    setCurrentEpoch(0);
    setProgress(0);
    setLogs([]);
  };

  useEffect(() => {
    return () => clearInterval(trainingInterval.current);
  }, []);

  return (
    <div className="flex w-full max-w-6xl flex-col">
      <Header />
      <main className="flex-grow py-12 md:py-20">
        <div className="max-w-7xl mx-auto">
          <PageTitle>Model Training Dashboard</PageTitle>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 flex flex-col gap-6">
              <Card>
                <h2 className="text-xl font-bold tracking-tight mb-5">Model Configuration</h2>
                <div className="flex flex-col gap-4">
                  <SelectInput label="Model Architecture" options={['Lightweight GTr', 'GATv2', 'Transformer-M']} />
                  <SelectInput label="Dataset" options={['BioPDB v1.2', 'Custom Dataset Alpha', 'PeptideNet Benchmark']} />
                </div>
                <div className="border-t border-border-light dark:border-border-dark my-6"></div>
                <h3 className="text-base font-semibold mb-4">Hyperparameters</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-1">
                      <TextInput label="Learning Rate" type="text" defaultValue="0.001" />
                  </div>
                  <div className="col-span-1">
                      <TextInput label="Batch Size" type="number" defaultValue="64" />
                  </div>
                  <div className="flex flex-col col-span-2">
                    <div className="flex justify-between items-center pb-2">
                      <p className="text-sm font-medium">Epochs</p>
                      <span className="text-sm font-semibold text-primary">{epochs}</span>
                    </div>
                    <input
                      className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer range-sm accent-primary"
                      max="500" min="10" type="range" value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value))}
                    />
                  </div>
                </div>
                <div className="border-t border-border-light dark:border-border-dark my-6"></div>
                <div className="flex items-center justify-between">
                  <label className="text-base font-medium cursor-pointer" htmlFor="early-stopping">Enable Early Stopping</label>
                  <div className="relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in">
                    <input defaultChecked className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white dark:bg-gray-800 border-4 border-slate-300 dark:border-gray-600 appearance-none cursor-pointer checked:right-0 checked:border-primary" id="early-stopping" name="toggle" type="checkbox" />
                    <label className="toggle-label block overflow-hidden h-6 rounded-full bg-slate-300 dark:bg-gray-600 cursor-pointer" htmlFor="early-stopping"></label>
                  </div>
                </div>
                <div className="border-t border-border-light dark:border-border-dark my-6"></div>
                <div className="flex gap-3">
                  <Button onClick={startTraining} disabled={status === 'Training...'}>
                    <span className="material-symbols-outlined">play_arrow</span>
                    Start Training
                  </Button>
                  <Button onClick={resetTraining} variant="icon">
                    <span className="material-symbols-outlined">refresh</span>
                  </Button>
                </div>
              </Card>
            </div>
            <div className="lg:col-span-2 flex flex-col gap-6">
              <Card>
                <h2 className="text-xl font-bold tracking-tight mb-4">Training Progress</h2>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="relative inline-flex items-center justify-center size-12">
                      <svg className="size-full" height="36" viewBox="0 0 36 36" width="36" xmlns="http://www.w3.org/2000/svg">
                        <circle className="stroke-current text-slate-200 dark:text-slate-700" cx="18" cy="18" fill="none" r="16" strokeWidth="3"></circle>
                        <circle
                          className="stroke-current text-primary" cx="18" cy="18" fill="none" r="16"
                          strokeDasharray="100" strokeDashoffset={100 - progress} strokeLinecap="round" strokeWidth="3"
                        ></circle>
                      </svg>
                      <span className="absolute text-sm font-semibold">{`${Math.round(progress)}%`}</span>
                    </div>
                    <div>
                      <p className="font-semibold">{`Epoch ${currentEpoch}/${epochs}`}</p>
                      <p className="text-sm text-text-light/70 dark:text-text-dark/70">{`Status: ${status}`}</p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button onClick={pauseTraining} disabled={status !== 'Training...' && status !== 'Paused'} variant="secondary">
                      <span className="material-symbols-outlined text-base">{status === 'Paused' ? 'play_arrow' : 'pause'}</span>
                      {status === 'Paused' ? 'Resume' : 'Pause'}
                    </Button>
                    <Button onClick={() => stopTraining()} disabled={status === 'Idle' || status === 'Completed'} variant="destructive">
                      <span className="material-symbols-outlined text-base">stop</span> Stop
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center md:text-left">
                  <MetricDisplay label="Current Loss" value="0.183" />
                  <MetricDisplay label="Validation Acc." value="92.4%" />
                  <MetricDisplay label="Elapsed Time" value="1h 12m" />
                  <MetricDisplay label="ETA" value="~2h 5m" />
                </div>
              </Card>
              <Card className="flex-1">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold tracking-tight">Training & Validation Loss</h2>
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-2"><span className="size-3 rounded-sm bg-primary"></span>Training</div>
                    <div className="flex items-center gap-2"><span className="size-3 rounded-sm bg-cyan-400"></span>Validation</div>
                  </div>
                </div>
                <div>
                  <img className="w-full h-auto object-cover rounded-lg aspect-[16/7]" alt="A line chart showing training and validation loss over epochs." src="https://lh3.googleusercontent.com/aida-public/AB6AXuCCOmZUFOVuwnLlsG_w-4qBhrKOPcfj9MaAIZh9kBZF33PCyaXlNxp_xEy2YxpenTNG2k76dzl-i48Ct1rghMo85emoKngPvKWIxwtv-Gn9F4tKVWgI4K6kCi1fHBYJl4eTBrYBvoiBmdOtfYKUoRMOIwZgM_-gVGjb8HbuxOCEte06CytWsw0C2Q976Bq_QH7oRSrUAzVCPcIdxn0aL50CCbKyh5UphBDSH-KLeBRADLfnX93Mb-Ubb2HfU4sTugkvoMhUJXWAztU" />
                </div>
              </Card>
            </div>
            <div className="lg:col-span-3">
              <Card>
                <div className="p-6 border-b border-border-light dark:border-border-dark">
                  <h2 className="text-xl font-bold tracking-tight">Console Log</h2>
                </div>
                <div className="p-6 bg-surface-dark/40 dark:bg-background-dark/80 rounded-b-xl h-64 overflow-y-auto">
                  <pre className="text-sm font-mono text-text-light/90 dark:text-text-dark/90 whitespace-pre-wrap">
                    <code className="language-bash">{logs.join('\n')}</code>
                  </pre>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </main>
      <footer className="mt-auto border-t border-slate-900/10 pt-8 pb-4 dark:border-slate-50/10">
        <div className="flex flex-col items-center justify-between gap-4 text-sm text-slate-500 dark:text-slate-400 sm:flex-row">
          <p>© 2024 AI Lab, University of Science. All rights reserved.</p>
          <div className="flex items-center gap-4">
            <a className="hover:text-primary" href="#">Contact</a>
            <a className="hover:text-primary" href="#">Privacy Policy</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Training;
