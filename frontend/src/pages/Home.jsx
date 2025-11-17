import React from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import ScrollAnimation from '../components/ScrollAnimation';
import Button from '../components/Button';
import Card from '../components/Card';
import HighlightCard from '../components/HighlightCard';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="flex w-full max-w-6xl flex-col">
      <Header />
      <main className="flex-grow py-12 md:py-20">
        {/* HeroSection */}
        <section className="text-center">
          <div className="mx-auto max-w-3xl">
            <h1 className="text-4xl font-black tracking-tighter sm:text-5xl md:text-6xl text-text-light dark:text-text-dark">LightGNN-Peptide</h1>
            <p className="mt-4 text-base text-slate-600 dark:text-slate-400 sm:text-lg md:text-xl">Stable Short Peptide Generation with Lightweight Graph Transformers</p>
            <div className="mt-8 flex justify-center">
              <Button onClick={() => navigate('/generation')} className="px-6">
                <span className="material-symbols-outlined">science</span>
                <span className="truncate">Start Generation</span>
              </Button>
            </div>
          </div>
        </section>
        {/* Workflow Visualization */}
        <ScrollAnimation>
          <section className="mt-16 md:mt-24">
            <h2 className="text-center text-2xl font-bold tracking-tight sm:text-3xl text-text-light dark:text-text-dark">Workflow Overview</h2>
            <Card className="mt-8 sm:p-8">
              <div className="flex flex-col items-center justify-between gap-6 md:flex-row md:gap-4">
                {/* Step 1: Input */}
                <div className="flex flex-col items-center text-center">
                  <div className="flex size-16 items-center justify-center rounded-full bg-primary/10 text-primary dark:bg-primary/20">
                    <span className="material-symbols-outlined text-3xl">database</span>
                  </div>
                  <h3 className="mt-4 text-lg font-bold">BioPDB Data Input</h3>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Protein structures from PDB</p>
                </div>
                {/* Arrow */}
                <div className="h-10 w-px rotate-90 bg-slate-300 dark:bg-slate-700 md:h-px md:w-16 md:rotate-0"></div>
                {/* Step 2: Model */}
                <div className="flex flex-col items-center text-center">
                  <div className="flex size-16 items-center justify-center rounded-full bg-primary/10 text-primary dark:bg-primary/20">
                    <span className="material-symbols-outlined text-3xl">hub</span>
                  </div>
                  <h3 className="mt-4 text-lg font-bold">Graph Transformer Model</h3>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Lightweight GNN processing</p>
                </div>
                {/* Arrow */}
                <div className="h-10 w-px rotate-90 bg-slate-300 dark:bg-slate-700 md:h-px md:w-16 md:rotate-0"></div>
                {/* Step 3: Output */}
                <div className="flex flex-col items-center text-center">
                  <div className="flex size-16 items-center justify-center rounded-full bg-primary/10 text-primary dark:bg-primary/20">
                    <span className="material-symbols-outlined text-3xl">biotech</span>
                  </div>
                  <h3 className="mt-4 text-lg font-bold">Stable Peptide Output</h3>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Novel peptide sequences</p>
                </div>
              </div>
            </Card>
          </section>
        </ScrollAnimation>
        {/* Project Highlights */}
        <ScrollAnimation>
          <section className="mt-16 md:mt-24">
            <h2 className="text-center text-2xl font-bold tracking-tight sm:text-3xl text-text-light dark:text-text-dark">Project Highlights</h2>
            <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
              <HighlightCard icon="model_training" title="Graph Transformer Architecture">
                Utilizes a lightweight yet powerful Graph Transformer model for efficient structural analysis and generation.
              </HighlightCard>
              <HighlightCard icon="data_object" title="Powered by BioPDB">
                Integrates seamlessly with the Protein Data Bank, leveraging a vast repository of biological macromolecular data.
              </HighlightCard>
              <HighlightCard icon="auto_awesome" title="Discovering Novel Peptides">
                Aims to accelerate the discovery of new, stable short peptides for therapeutic and industrial applications.
              </HighlightCard>
            </div>
          </section>
        </ScrollAnimation>
      </main>
      {/* Footer */}
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

export default Home;
