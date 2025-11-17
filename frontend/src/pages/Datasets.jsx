import React, { useState, useRef } from 'react';
import Card from '../components/Card';
import Button from '../components/Button';
import DatasetListItem from '../components/DatasetListItem';

const Datasets = () => {
  const [datasets, setDatasets] = useState([
    { name: 'peptide_model_alpha.pdb', date: '2023-10-27', status: 'Analyzed' },
    { name: 'complex_structure_beta.pdb', date: '2023-10-26', status: 'Processing' },
    { name: 'inhibitor_target_gamma.pdb', date: '2023-10-25', status: 'Raw' },
  ]);
  const [selectedDataset, setSelectedDataset] = useState(datasets[0]);
  const fileInputRef = useRef(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const newDataset = {
        name: file.name,
        date: new Date().toISOString().slice(0, 10),
        status: 'Processing',
      };
      setDatasets((prev) => [newDataset, ...prev]);
      setSelectedDataset(newDataset);

      // Simulate processing
      setTimeout(() => {
        setDatasets((prev) =>
          prev.map((d) =>
            d.name === newDataset.name ? { ...d, status: 'Analyzed' } : d
          )
        );
      }, 3000);
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current.click();
  };

  return (
    <>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 w-full">
        {/* Left Column */}
          <div className="lg:col-span-1 flex flex-col gap-8">
            <Card>
              <div className="p-6">
                <h1 className="text-[#0d141b] dark:text-white tracking-light text-xl font-bold leading-tight text-left">Dataset Management</h1>
              </div>
              <div className="flex flex-col p-6 pt-0">
                <div className="flex flex-col items-center gap-6 rounded-lg border-2 border-dashed border-[#cfdbe7] dark:border-slate-700 px-6 py-14">
                  <div className="flex max-w-[480px] flex-col items-center gap-2">
                    <p className="text-[#0d141b] dark:text-white text-lg font-bold leading-tight tracking-[-0.015em] max-w-[480px] text-center">Drag and drop PDB files</p>
                    <p className="text-slate-600 dark:text-slate-400 text-sm font-normal leading-normal max-w-[480px] text-center">Accepted file types: .pdb. Max file size: 100MB.</p>
                  </div>
                  <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept=".pdb" style={{ display: 'none' }} />
                  <Button onClick={triggerFileUpload} variant="secondary" className="h-10 px-4">
                    Upload PDB File
                  </Button>
                </div>
              </div>
            </Card>
            <Card className="flex-1 flex flex-col">
              <div className="p-6">
                <h2 className="text-[#0d141b] dark:text-white text-lg font-bold leading-tight tracking-[-0.015em]">Uploaded Datasets</h2>
              </div>
              <div className="flex-1 px-2 pb-4 overflow-y-auto">
                <div className="flex flex-col gap-1">
                  {datasets.map((dataset) => (
                    <DatasetListItem
                      key={dataset.name}
                      dataset={dataset}
                      selectedDataset={selectedDataset}
                      setSelectedDataset={setSelectedDataset}
                    />
                  ))}
                </div>
              </div>
            </Card>
          </div>
          {/* Right Column */}
          <div className="lg:col-span-2 flex flex-col gap-8">
            <Card className="p-6 flex-col flex">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-[#0d141b] dark:text-white text-xl font-bold leading-tight tracking-[-0.015em]">Analysis & Visualization</h2>
                <Button className="h-10 px-4">
                  <span className="material-symbols-outlined text-lg">play_arrow</span>
                  Process Dataset
                </Button>
              </div>
              <Card className="flex-1 min-h-[400px]">
                <h3 className="text-[#0d141b] dark:text-white font-bold text-lg mb-2">3D Structure Viewer</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">Viewing: {selectedDataset.name}</p>
                <div className="bg-slate-200/50 dark:bg-slate-800/50 rounded-lg h-full flex items-center justify-center relative overflow-hidden">
                  <img alt="3D rendering of a peptide molecular structure" className="w-full h-full object-contain" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCLTFomW4Mqc_mcSAQpXhex8AjVGd-1QGbN2EaHV-2ogtPtdRvl61kYXHHY_hSqp9QU0Nx5fceeYuRSIalgS66iU3Bds1PMqOBlPLOqzU7quZRDZB9JiAZ-LouGF00AQ0j3Oymcb6nNX361WTZQcwopugEILjJRHMH7k650wFW-iYJZmd22b1XEjvxpVkJeP1F6_pIv_NlsdxCqIbePfmXLw0cAZ6Tu0ndky5bEVI8UzSTs21JYydUte42piXZZjD4GO3cYqiThFkw" />
                  <div className="absolute bottom-4 right-4 flex gap-2">
                    <Button variant="icon" className="size-9 bg-white/70 dark:bg-slate-900/70 shadow-md">
                      <span className="material-symbols-outlined text-lg">zoom_in</span>
                    </Button>
                    <Button variant="icon" className="size-9 bg-white/70 dark:bg-slate-900/70 shadow-md">
                      <span className="material-symbols-outlined text-lg">zoom_out</span>
                    </Button>
                    <Button variant="icon" className="size-9 bg-white/70 dark:bg-slate-900/70 shadow-md">
                      <span className="material-symbols-outlined text-lg">360</span>
                    </Button>
                  </div>
                </div>
              </Card>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
                <Card className="p-4">
                  <h4 className="text-base font-bold text-[#0d141b] dark:text-white mb-2">Structural Properties</h4>
                  <div className="space-y-2 text-sm text-slate-600 dark:text-slate-300">
                    <div className="flex justify-between"><span>Atom Count:</span> <span className="font-medium text-[#0d141b] dark:text-white">1,204</span></div>
                    <div className="flex justify-between"><span>Bond Count:</span> <span className="font-medium text-[#0d141b] dark:text-white">1,238</span></div>
                    <div className="flex justify-between"><span>Residues:</span> <span className="font-medium text-[#0d141b] dark:text-white">78</span></div>
                  </div>
                </Card>
                <Card className="p-4">
                  <h4 className="text-base font-bold text-[#0d141b] dark:text-white mb-2">Graph Metrics</h4>
                  <div className="space-y-2 text-sm text-slate-600 dark:text-slate-300">
                    <div className="flex justify-between"><span>Node Count:</span> <span className="font-medium text-[#0d141b] dark:text-white">1,204</span></div>
                    <div className="flex justify-between"><span>Edge Count:</span> <span className="font-medium text-[#0d141b] dark:text-white">2,476</span></div>
                    <div className="flex justify-between"><span>Density:</span> <span className="font-medium text-[#0d141b] dark:text-white">0.0017</span></div>
                  </div>
                </Card>
                <Card className="p-4 flex flex-col justify-center items-center">
                  <h4 className="text-base font-bold text-[#0d141b] dark:text-white mb-2">Model Readiness</h4>
                  <div className="flex items-center gap-2 text-green-700 dark:text-green-400 font-bold">
                    <span className="material-symbols-outlined">check_circle</span>
                    <span>Ready</span>
                  </div>
                </Card>
              </div>
              <Card className="mt-6 p-4">
                <h4 className="text-base font-bold text-[#0d141b] dark:text-white mb-2">Node Degree Distribution</h4>
                <div className="w-full h-40 flex items-center justify-center">
                  <img alt="A line chart showing the node degree distribution" className="w-full h-full object-contain" src="https://lh3.googleusercontent.com/aida-public/AB6AXuBEaFcpknaPuzA_N2GbqyByTtm_PXCuSZIA42kl4V3x9LPbs6p4qdJ17Ly96VT03yw3RaN5Cn9DB16ZdY-ub5H9d3MiqnU4BNvkm0Z45a7Vywp6bsV9JY7n-OyRlbBqqUZIsPMX7Ki5X9sqJ6HdFyKAbcA_u7NuY2PzPs0ARy4eqodzjwB7sGQN8jbwcggj-AnZXTMfETeLFsdR7RDsa6s0oCgRRuxr46qto1-CUj3Ly7_tldH1xXfVmChZmBImCWEC25X1NjzrI5s" />
                </div>
              </Card>
            </Card>
          </div>
        </div>
    </>
  );
};

export default Datasets;
