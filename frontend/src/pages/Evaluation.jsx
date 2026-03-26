import React from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import Card from '../components/Card';
import MetricCard from '../components/MetricCard';

const Evaluation = () => {
  const { t } = useTranslation();
  const base = '/Peptide-Design-Lightweight-Model/';

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5, ease: 'easeOut' }
    }
  };

  const peptideData = [
    { sequence: 'IGKLFKLTQ', stability: 35.4, amp: 0.98, status: 'Stable' },
    { sequence: 'GKVFWCT', stability: 30.8, amp: 0.92, status: 'Stable' },
    { sequence: 'YVGDKLFEILGGRYGGD', stability: 22.1, amp: 0.88, status: 'Stable' },
    { sequence: 'WIPPRGAFF', stability: 39.1, amp: 0.95, status: 'Stable' },
  ];

  return (
    <motion.div 
      className="layout-content-container flex flex-col flex-1 gap-10 pb-20"
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      {/* Header Section */}
      <motion.div variants={itemVariants} className="flex flex-col gap-2">
        <h1 className="text-4xl font-black text-slate-900 dark:text-white tracking-tight">
          Model <span className="text-primary italic">Assessment</span>
        </h1>
        <p className="text-slate-500 dark:text-slate-400 text-lg max-w-3xl">
          Comprehensive evaluation of the LightweightPeptideModel, highlighting its biological efficacy, 
          structural stability, and architectural superiority over baseline models.
        </p>
      </motion.div>

      {/* Baseline Comparison Section */}
      <motion.div variants={itemVariants} className="flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">analytics</span>
            Baseline Performance Analysis
          </h2>
          <p className="text-slate-500 dark:text-slate-400">
            So sánh chi tiết hiệu năng của LightweightPeptideGen với các mô hình Baseline (ESM2Gen, HydrAMP, M3CAD, PepGraphormer) 
            trên 5 tiêu chí đánh giá cốt lõi.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Entropy */}
          <Card className="flex-col gap-4 p-5 glass-morphism">
            <div className="flex justify-between items-start">
              <h3 className="font-bold text-slate-800 dark:text-white uppercase tracking-wider text-sm flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-500"></span> Entropy
              </h3>
            </div>
            <div className="rounded-xl overflow-hidden bg-white/50 dark:bg-black/20 border border-slate-100 dark:border-white/5 p-2">
              <img src={`${base}entropy.png`} alt="Entropy Chart" className="w-full h-auto" />
            </div>
            <p className="text-xs text-slate-500 leading-relaxed italic">
              Đo lường độ bất định và sự phong phú của amino acid. Chỉ số cao cho thấy mô hình sinh ra chuỗi có cấu trúc phức tạp, không bị lặp lại đơn điệu.
            </p>
          </Card>

          {/* Bigram Diversity */}
          <Card className="flex-col gap-4 p-5 glass-morphism">
             <h3 className="font-bold text-slate-800 dark:text-white uppercase tracking-wider text-sm flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-orange-500"></span> Bigram Diversity
              </h3>
            <div className="rounded-xl overflow-hidden bg-white/50 dark:bg-black/20 border border-slate-100 dark:border-white/5 p-2">
              <img src={`${base}ngram_diversity_2.png`} alt="Bigram Diversity" className="w-full h-auto" />
            </div>
            <p className="text-xs text-slate-500 leading-relaxed italic">
              Đánh giá độ đa dạng của các cặp amino acid kế tiếp. Giúp xác nhận mô hình học được các pattern motif sinh học đa dạng thay vì chỉ một vài tổ hợp cố định.
            </p>
          </Card>

          {/* Uniqueness Ratio */}
          <Card className="flex-col gap-4 p-5 glass-morphism">
             <h3 className="font-bold text-slate-800 dark:text-white uppercase tracking-wider text-sm flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Uniqueness Ratio
              </h3>
            <div className="rounded-xl overflow-hidden bg-white/50 dark:bg-black/20 border border-slate-100 dark:border-white/5 p-2">
              <img src={`${base}uniqueness_ratio.png`} alt="Uniqueness Ratio" className="w-full h-auto" />
            </div>
            <p className="text-xs text-slate-500 leading-relaxed italic">
              Tỷ lệ các chuỗi duy nhất trong tập kết quả. LightweightPeptideGen đạt mức gần như 1.0, chứng tỏ khả năng sáng tạo chuỗi mới cực tốt.
            </p>
          </Card>

          {/* Stable Ratio */}
          <Card className="flex-col gap-4 p-5 glass-morphism">
             <h3 className="font-bold text-slate-800 dark:text-white uppercase tracking-wider text-sm flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-rose-500"></span> Stable Ratio
              </h3>
            <div className="rounded-xl overflow-hidden bg-white/50 dark:bg-black/20 border border-slate-100 dark:border-white/5 p-2">
              <img src={`${base}stable_ratio.png`} alt="Stable Ratio" className="w-full h-auto" />
            </div>
            <p className="text-xs text-slate-500 leading-relaxed italic">
              Tỷ lệ chuỗi có chỉ số Instability Index {'<'} 40. Đây là minh chứng cho việc mô hình không chỉ quan tâm đến hoạt tính mà còn đảm bảo Peptide bền vững về mặt cấu trúc.
            </p>
          </Card>

          {/* Mean Length */}
          <Card className="flex-col gap-4 p-5 glass-morphism">
             <h3 className="font-bold text-slate-800 dark:text-white uppercase tracking-wider text-sm flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-500"></span> Mean Length
              </h3>
            <div className="rounded-xl overflow-hidden bg-white/50 dark:bg-black/20 border border-slate-100 dark:border-white/5 p-2">
              <img src={`${base}mean_length.png`} alt="Mean Length" className="w-full h-auto" />
            </div>
            <p className="text-xs text-slate-500 leading-relaxed italic">
              Chiều dài trung bình của Peptide. Mô hình của chúng tôi tối ưu chiều dài ngắn hơn (khoảng 14-15 AA) giúp giảm chi phí tổng hợp trong thực tế.
            </p>
          </Card>
        </div>
      </motion.div>

      {/* Top Level Metrics */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard title="Avg. AMP Probability" value="92.4%" delta="+8.5%" />
        <MetricCard title="Stability Rate (II<40)" value="70.5%" delta="+12.2%" />
        <MetricCard title="Sequence Uniqueness" value="100%" delta="Stable" />
        <MetricCard title="Diversity Index" value="0.644" delta="+0.05" />
      </motion.div>

      {/* Biological & Chemical Properties Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Bio-Efficacy Section */}
        <motion.div variants={itemVariants} className="flex flex-col gap-6">
          <div className="flex items-center gap-3 px-2">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-500">
              <span className="material-symbols-outlined">biotech</span>
            </div>
            <h3 className="text-xl font-bold dark:text-white">Biological Efficacy</h3>
          </div>
          
          <Card className="flex-col gap-4 p-6 glass-morphism">
            <h4 className="font-bold text-slate-800 dark:text-slate-200">AMP Probability Distribution</h4>
            <div className="rounded-xl overflow-hidden bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-800">
              <img src={`${base}amp_probability_hist.png`} alt="AMP Probability" className="w-full" />
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400 italic">
              Measures the model's ability to generate sequences with high antimicrobial potential ({'>'}0.8).
            </p>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="flex-col gap-4 p-5 glass-morphism text-center">
              <h4 className="font-bold text-sm text-slate-800 dark:text-slate-200 mb-2">Hemolytic Score</h4>
              <img src={`${base}hemolytic_score_hist.png`} alt="Hemolytic Score" className="rounded-lg w-full" />
            </Card>
            <Card className="flex-col gap-4 p-5 glass-morphism text-center">
              <h4 className="font-bold text-sm text-slate-800 dark:text-slate-200 mb-2">Therapeutic Index</h4>
              <img src={`${base}therapeutic_score_hist.png`} alt="Therapeutic Score" className="rounded-lg w-full" />
            </Card>
          </div>
        </motion.div>

        {/* Structural Stability Section */}
        <motion.div variants={itemVariants} className="flex flex-col gap-6">
          <div className="flex items-center gap-3 px-2">
            <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-500">
              <span className="material-symbols-outlined">architecture</span>
            </div>
            <h3 className="text-xl font-bold dark:text-white">Structural Stability</h3>
          </div>

          <Card className="flex-col gap-4 p-6 glass-morphism">
            <h4 className="font-bold text-slate-800 dark:text-slate-200">Instability Index Distribution</h4>
            <div className="rounded-xl overflow-hidden bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-800">
              <img src={`${base}instability_hist.png`} alt="Instability Index" className="w-full" />
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400 italic">
              Statistical summary of sequence longevity. Lower values ({'<'}40) indicate higher structural integrity.
            </p>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="flex-col gap-4 p-5 glass-morphism text-center">
              <h4 className="font-bold text-sm text-slate-800 dark:text-slate-200 mb-2">GRAVY Score</h4>
              <img src={`${base}gravy_hist.png`} alt="GRAVY Score" className="rounded-lg w-full" />
            </Card>
            <Card className="flex-col gap-4 p-5 glass-morphism text-center">
              <h4 className="font-bold text-sm text-slate-800 dark:text-slate-200 mb-2">Sequence Length</h4>
              <img src={`${base}length_hist.png`} alt="Length Distribution" className="rounded-lg w-full" />
            </Card>
          </div>
        </motion.div>
      </div>

      {/* Distribution & Correlations */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <Card className="flex-col gap-6 p-6 glass-morphism h-full">
            <h3 className="text-xl font-bold dark:text-white flex items-center gap-2">
              <span className="material-symbols-outlined text-purple-500">palette</span>
              Amino Acid Composition
            </h3>
            <div className="flex-1 flex items-center justify-center bg-slate-50 dark:bg-slate-900/50 rounded-2xl border border-slate-100 dark:border-slate-800 overflow-hidden">
              <img src={`${base}aa_distribution.png`} alt="AA Distribution" className="w-full object-contain" />
            </div>
          </Card>
        </motion.div>

        <motion.div variants={itemVariants} className="flex flex-col gap-6">
          <Card className="flex-col gap-4 p-5 glass-morphism flex-1">
             <h4 className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
               <span className="material-symbols-outlined text-rose-500">grid_guides</span> AMP vs Hemolytic
             </h4>
             <img src={`${base}amp_vs_hemolytic.png`} alt="Correlation Plot" className="rounded-lg shadow-sm w-full" />
          </Card>
          <Card className="flex-col gap-4 p-5 glass-morphism flex-1">
             <h4 className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
               <span className="material-symbols-outlined text-amber-500">blur_on</span> Stability vs Gravy
             </h4>
             <img src={`${base}instability_vs_gravy.png`} alt="Correlation Plot 2" className="rounded-lg shadow-sm w-full" />
          </Card>
        </motion.div>
      </div>

      {/* Samples Table */}
      <motion.div variants={itemVariants}>
        <Card className="flex-col gap-6 p-6 glass-morphism overflow-hidden">
          <h3 className="text-xl font-bold dark:text-white">Validation Sample Subset</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800">
                  <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">Sequence</th>
                  <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">Instability</th>
                  <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">AMP Prob.</th>
                  <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">State</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {peptideData.map((peptide, index) => (
                  <tr key={index} className="hover:bg-slate-50 dark:hover:bg-white/5 transition-colors group">
                    <td className="p-4 text-sm font-mono text-slate-800 dark:text-slate-200 font-bold group-hover:text-primary transition-colors">
                      {peptide.sequence}
                    </td>
                    <td className="p-4 text-sm text-slate-600 dark:text-slate-400">{peptide.stability}</td>
                    <td className="p-4 text-sm font-bold text-emerald-600 dark:text-emerald-400 tracking-wider">
                      {(peptide.amp * 100).toFixed(1)}%
                    </td>
                    <td className="p-4">
                      <span className="px-3 py-1 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 text-[10px] font-black uppercase rounded-full tracking-widest">
                        {peptide.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </motion.div>

      <style dangerouslySetInnerHTML={{ __html: `
        .glass-morphism {
          background: rgba(255, 255, 255, 0.7);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.4);
        }
        .dark .glass-morphism {
          background: rgba(15, 23, 42, 0.7);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.05);
        }
      `}} />
    </motion.div>
  );
};

export default Evaluation;
