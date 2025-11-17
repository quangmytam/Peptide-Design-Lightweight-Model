import React, { useState } from 'react';
import Header from '../components/Header';
import TextInput from '../components/TextInput';
import SelectInput from '../components/SelectInput';
import Button from '../components/Button';
import PeptideResultCard from '../components/PeptideResultCard';

const Generation = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedPeptides, setGeneratedPeptides] = useState([]);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setGeneratedPeptides([]);

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const mockPeptides = [
      { sequence: 'ACDEFGHIK', stability: 0.92 },
      { sequence: 'LMNPQRSTV', stability: 0.88 },
      { sequence: 'WYACDEFGH', stability: 0.85 },
    ];

    setGeneratedPeptides(mockPeptides);
    setIsGenerating(false);
  };

  return (
    <div className="flex w-full max-w-6xl flex-col">
      <Header />
      <main className="flex-grow py-12 md:py-20">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 w-full">
          {/* Left Column: Control Panel */}
          <aside className="lg:col-span-4 xl:col-span-3 flex flex-col gap-8">
            <div className="flex flex-col gap-2">
              <h1 className="text-4xl font-black tracking-[-0.033em]">Peptide Generation</h1>
              <p className="text-base font-normal leading-normal text-subtext-light dark:text-subtext-dark">Configure parameters and generate stable short peptides.</p>
            </div>
            <div className="space-y-6">
              <h2 className="text-xl font-bold tracking-[-0.015em] border-b border-border-light dark:border-border-dark pb-3">Generation Parameters</h2>
              <TextInput label="Target Protein ID" placeholder="e.g., PDB:1A2B" />
              <TextInput label="Desired Peptide Length" type="number" placeholder="e.g., 10" />
              <SelectInput label="Graph Transformer Model" options={['LightGNN-v2 (Recommended)', 'LightGNN-v1', 'Transformer-XL']} />
              <SelectInput label="BioPDB Dataset Version" options={['BioPDB 2024-Q2', 'BioPDB 2024-Q1', 'BioPDB 2023-Q4']} />
              <div className="flex items-center justify-between pt-2">
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium" htmlFor="stability-filter">Enable Stability Filter</label>
                  <span className="material-symbols-outlined text-base text-subtext-light dark:text-subtext-dark cursor-help" title="Only return peptides with a predicted stability score above 0.85.">help</span>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input defaultChecked className="sr-only peer" id="stability-filter" type="checkbox" />
                  <div className="w-11 h-6 bg-gray-200 dark:bg-gray-700 rounded-full peer peer-focus:ring-2 peer-focus:ring-primary/50 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-primary"></div>
                </label>
              </div>
              <div className="pt-4 border-t border-border-light dark:border-border-dark">
                <Button onClick={handleGenerate} disabled={isGenerating} className="w-full">
                  <span className="material-symbols-outlined">auto_awesome</span>
                  {isGenerating ? 'Generating...' : 'Generate Peptides'}
                </Button>
              </div>
              {isGenerating && (
                <div className="flex flex-col items-center gap-3 p-4 rounded-lg bg-primary/10 dark:bg-primary/20">
                  <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
                  <p className="text-sm font-medium text-primary">Generation in progress... (Est. 45s)</p>
                </div>
              )}
            </div>
          </aside>
          {/* Right Column: Results Area */}
          <section className="lg:col-span-8 xl:col-span-9 flex flex-col gap-6">
            <div className="flex flex-wrap justify-between items-center gap-3">
              <h2 className="text-2xl font-bold tracking-[-0.015em]">Generated Peptides</h2>
              <p className="text-sm font-medium text-subtext-light dark:text-subtext-dark">
                {generatedPeptides.length > 0 ? `Showing ${generatedPeptides.length} results` : ''}
              </p>
            </div>
            <div className="grid grid-cols-1 @container md:grid-cols-2 xl:grid-cols-3 gap-6">
              {generatedPeptides.length > 0 ? (
                generatedPeptides.map((peptide, index) => (
                  <PeptideResultCard key={index} peptide={peptide} index={index} />
                ))
              ) : (
                <div className="lg:col-span-3 flex-1 flex flex-col items-center justify-center text-center p-10 border-2 border-dashed border-border-light dark:border-border-dark rounded-xl">
                  <span className="material-symbols-outlined text-6xl text-subtext-light dark:text-subtext-dark opacity-50">science</span>
                  <h3 className="text-xl font-bold mt-4">No Peptides Generated Yet</h3>
                  <p className="text-subtext-light dark:text-subtext-dark mt-2 max-w-sm">Use the control panel on the left to configure your parameters and start a new generation process. Your results will appear here.</p>
                </div>
              )}
            </div>
          </section>
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

export default Generation;
