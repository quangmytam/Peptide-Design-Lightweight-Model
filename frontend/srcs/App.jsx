import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import PageLayout from './components/PageLayout';
import Home from './pages/Home';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import Generation from './pages/Generation';
import Datasets from './pages/Datasets';

function App() {
  return (
    <BrowserRouter basename="/LightGNN-Peptide">
      <Routes>
        <Route element={<PageLayout />}>
          <Route index element={<Home />} />
          <Route path="training" element={<Training />} />
          <Route path="evaluation" element={<Evaluation />} />
          <Route path="generation" element={<Generation />} />
          <Route path="datasets" element={<Datasets />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
