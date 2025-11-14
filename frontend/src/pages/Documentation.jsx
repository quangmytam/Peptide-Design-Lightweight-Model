import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Header from '../components/Header';
const Documentation = () => {
    const [filters, setFilters] = useState({
      modelVersion: 0,
      dataset: 0,
      date: 0,
    });
  
    const modelVersions = ['v1.2', 'v1.1', 'v1.0'];
    const datasets = ['BioPDB', 'Custom Alpha', 'PeptideNet'];
    const dates = ['Latest', 'Last 7 Days', 'Last 30 Days'];
  
    const handleFilterClick = (filterName) => {
      setFilters((prev) => ({
        ...prev,
        [filterName]: (prev[filterName] + 1) % (
          filterName === 'modelVersion' ? modelVersions.length :
          filterName === 'dataset' ? datasets.length : dates.length
        ),
      }));
    };
  
    const navLinks = [
      { to: '/datasets', label: 'Datasets' },
      { to: '/training', label: 'Training' },
      { to: '/generation', label: 'Generation' },
      { to: '/evaluation', label: 'Evaluation' },
      { to: '/about', label: 'About' },
    ];
  return (
    <div className="p-10">
      <h1 className="text-4xl font-black">Documentation</h1>
      <p className="mt-4">
        This is the documentation page. Information about the LightGNN-Peptide system will be provided here.
      </p>
    </div>
  );
};

export default Documentation;
