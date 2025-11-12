import React, { useState } from 'react';

const Evaluation = () => {
  const [modelVersion, setModelVersion] = useState('v1.2');
  const [dataset, setDataset] = useState('BioPDB');

  return (
    <div className="layout-content-container flex flex-col flex-1 gap-8">
      <div className="flex flex-wrap justify-between gap-4 items-start">
        <div className="flex min-w-72 flex-col gap-2">
          <p className="text-slate-900 dark:text-slate-50 text-4xl font-black leading-tight tracking-[-0.033em]">LightGNN-Peptide: Model Evaluation</p>
          <p className="text-slate-500 dark:text-slate-400 text-base font-normal leading-normal">Performance metrics for the peptide generation model trained on the BioPDB dataset.</p>
        </div>
      </div>
      <div className="flex gap-3 overflow-x-auto pb-2">
        <div className="dropdown">
          <button className="flex h-8 shrink-0 items-center justify-center gap-x-2 rounded-lg bg-slate-200 dark:bg-slate-800 px-4">
            <p className="text-slate-900 dark:text-slate-50 text-sm font-medium leading-normal">Model Version: {modelVersion}</p>
            <span className="material-symbols-outlined text-slate-500 dark:text-slate-400">expand_more</span>
          </button>
          <div className="dropdown-content">
            <a href="#" onClick={() => setModelVersion('v1.2')}>v1.2</a>
            <a href="#" onClick={() => setModelVersion('v1.1')}>v1.1</a>
            <a href="#" onClick={() => setModelVersion('v1.0')}>v1.0</a>
          </div>
        </div>
        <div className="dropdown">
          <button className="flex h-8 shrink-0 items-center justify-center gap-x-2 rounded-lg bg-slate-200 dark:bg-slate-800 px-4">
            <p className="text-slate-900 dark:text-slate-50 text-sm font-medium leading-normal">Dataset: {dataset}</p>
            <span className="material-symbols-outlined text-slate-500 dark:text-slate-400">expand_more</span>
          </button>
          <div className="dropdown-content">
            <a href="#" onClick={() => setDataset('BioPDB')}>BioPDB</a>
            <a href="#" onClick={() => setDataset('Custom Dataset Alpha')}>Custom Dataset Alpha</a>
            <a href="#" onClick={() => setDataset('PeptideNet Benchmark')}>PeptideNet Benchmark</a>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="flex flex-col gap-2 rounded-xl p-6 border border-slate-200 dark:border-slate-800 bg-white/70 dark:bg-slate-900/70 backdrop-blur-sm">
            <p className="text-slate-700 dark:text-slate-300 text-base font-medium leading-normal">Overall Validity</p>
            <p className="text-slate-900 dark:text-slate-50 tracking-tight text-3xl font-bold leading-tight">98.2%</p>
            <p className="text-green-600 dark:text-green-500 text-sm font-medium leading-normal">+1.5%</p>
        </div>
        <div className="flex flex-col gap-2 rounded-xl p-6 border border-slate-200 dark:border-slate-800 bg-white/70 dark:bg-slate-900/70 backdrop-blur-sm">
            <p className="text-slate-700 dark:text-slate-300 text-base font-medium leading-normal">Average Stability</p>
            <p className="text-slate-900 dark:text-slate-50 tracking-tight text-3xl font-bold leading-tight">0.91</p>
            <p className="text-green-600 dark:text-green-500 text-sm font-medium leading-normal">+0.02</p>
        </div>
        <div className="flex flex-col gap-2 rounded-xl p-6 border border-slate-200 dark:border-slate-800 bg-white/70 dark:bg-slate-900/70 backdrop-blur-sm">
            <p className="text-slate-700 dark:text-slate-300 text-base font-medium leading-normal">Uniqueness</p>
            <p className="text-slate-900 dark:text-slate-50 tracking-tight text-3xl font-bold leading-tight">99.7%</p>
            <p className="text-red-600 dark:text-red-500 text-sm font-medium leading-normal">-0.1%</p>
        </div>
        <div className="flex flex-col gap-2 rounded-xl p-6 border border-slate-200 dark:border-slate-800 bg-white/70 dark:bg-slate-900/70 backdrop-blur-sm">
            <p className="text-slate-700 dark:text-slate-300 text-base font-medium leading-normal">Diversity Score</p>
            <p className="text-slate-900 dark:text-slate-50 tracking-tight text-3xl font-bold leading-tight">0.85</p>
            <p className="text-green-600 dark:text-green-500 text-sm font-medium leading-normal">+0.04</p>
        </div>
    </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="flex flex-col gap-4 rounded-xl border border-slate-200 dark:border-slate-800 p-6 bg-white/70 dark:bg-slate-900/70 backdrop-blur-sm">
          {/* ... (Performance Metrics chart) ... */}
        </div>
        <div className="flex flex-col gap-4 rounded-xl border border-slate-200 dark:border-slate-800 p-6 bg-white/70 dark:bg-slate-900/70 backdrop-blur-sm">
          {/* ... (Training Progression chart) ... */}
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-2 flex flex-col gap-4 rounded-xl border border-slate-200 dark:border-slate-800 p-6 bg-white/70 dark:bg-slate-900/70 backdrop-blur-sm">
          {/* ... (Overall Model Performance chart) ... */}
        </div>
        <div className="lg:col-span-3 flex flex-col gap-4 rounded-xl border border-slate-200 dark:border-slate-800 p-6 bg-white/70 dark:bg-slate-900/70 backdrop-blur-sm">
          <div className="flex flex-col">
            <p className="text-slate-900 dark:text-slate-50 text-lg font-bold leading-normal">Sample Generated Peptides</p>
            <p className="text-slate-500 dark:text-slate-400 text-sm font-normal leading-normal">A list of generated peptides with their scores</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800">
                  <th className="p-3 text-sm font-semibold text-slate-600 dark:text-slate-400">Sequence</th>
                  <th className="p-3 text-sm font-semibold text-slate-600 dark:text-slate-400">Stability</th>
                  <th className="p-3 text-sm font-semibold text-slate-600 dark:text-slate-400">Validity</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                <tr>
                  <td className="p-3 text-sm font-mono text-slate-800 dark:text-slate-200">ACDEFGHIKL</td>
                  <td className="p-3 text-sm text-slate-600 dark:text-slate-300">0.95</td>
                  <td className="p-3 text-sm text-green-600 dark:text-green-500">Valid</td>
                </tr>
                {/* ... (more rows) ... */}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Evaluation;
