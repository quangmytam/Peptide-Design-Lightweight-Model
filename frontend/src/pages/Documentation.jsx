import React, { useRef } from 'react';

const slidestyle = "bg-white/50 dark:bg-background-dark/50 backdrop-blur-lg rounded-3xl px-6 md:px-10 lg:px-16 py-8 md:py-10 lg:py-12 shadow-xl border border-gray-200/80 dark:border-gray-700/80 relative transition-all duration-300 w-[97%] max-w-[1900px] mx-auto flex flex-col min-h-[85vh] my-[4vh] group cursor-pointer overflow-hidden";

const slidesData = [
  {
    id: 1,
    title: "",
    content: (
      <div className="text-center mt-4 space-y-6">
        <img
          src="/Peptide-Design-Lightweight-Model/Logo_VLU_2022.png"
          alt="Logo Trường Đại học Văn Lang"
          className="mx-auto h-14 md:h-16 object-contain drop-shadow-md mb-2"
        />
        <p className="text-xl text-gray-500 dark:text-gray-400 font-bold uppercase tracking-widest mb-2">BÁO CÁO ĐỒ ÁN TỐT NGHIỆP</p>
        <p className="text-lg font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-widest mb-6 border-b-2 border-blue-200 dark:border-blue-900 inline-block pb-2">Khoa Công Nghệ Thông Tin - Chuyên ngành Trí Tuệ Nhân Tạo</p>
        <h2 className="text-4xl lg:text-[2.5rem] font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 mb-8 leading-snug min-h-[8rem] md:min-h-[10rem] flex items-center justify-center py-4 uppercase">
          Nghiên cứu kiến trúc LightweightPeptideGen cho sinh tạo chuỗi Peptide ngắn có độ ổn định và chức năng cao
        </h2>
        <div className="text-xl font-medium text-slate-800 dark:text-gray-200 space-y-4 max-w-xl mx-auto bg-white/60 dark:bg-slate-800/60 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm relative z-10">
          <p className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2"><strong>Sinh viên thực hiện:</strong> <span>Quang Mỹ Tâm - 2274802010784</span></p>
          <p className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2"><strong>Giảng viên hướng dẫn:</strong> <span className="text-primary font-bold">ThS. Nguyễn Thị Mỹ Linh</span></p>
        </div>
      </div>
    )
  },
  {
    id: 2,
    title: "1. Peptide và Amino Acid — Nền tảng sinh học của bài toán",
    content: (
      <div className="grid md:grid-cols-2 gap-10 text-xl items-center h-full">
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-8 rounded-3xl border border-blue-200 dark:border-blue-800 shadow-sm relative">
            <h3 className="font-bold text-2xl text-blue-600 dark:text-blue-400 mb-4 flex items-center gap-2 border-b-2 border-blue-200 dark:border-blue-800 pb-3">
              <span className="material-symbols-outlined text-3xl">biotech</span> Cơ sở Peptide
            </h3>
            <p className="font-medium text-slate-700 dark:text-slate-300 leading-relaxed">
              <strong>Peptide</strong> là phân tử sinh học gồm <strong>2–50 amino acid</strong> nối nhau bằng liên kết –CO–NH–.
            </p>
            <ul className="list-disc pl-6 mt-4 space-y-3 text-lg font-medium">
              <li>Amino acid là "token sinh học" của tự nhiên (20 loại tiêu chuẩn).</li>
              <li>Peptide kháng khuẩn (AMPs) phá vỡ màng tế bào vi khuẩn.</li>
              <li>Đặc thù: Chuỗi + Cấu trúc rỗng 3D + Năng lượng sinh học.</li>
            </ul>
          </div>
        </div>
        <div className="space-y-6">
          <div className="bg-rose-50 dark:bg-rose-900/10 p-8 rounded-3xl border border-rose-200 dark:border-rose-800 shadow-sm">
            <h3 className="font-bold text-2xl text-rose-600 dark:text-rose-400 mb-4 flex items-center gap-2 border-b-2 border-rose-200 dark:border-rose-800 pb-3">
              <span className="material-symbols-outlined text-3xl">public</span> Không gian tổ hợp vô tận
            </h3>
            <div className="text-center my-6">
              <p className="text-lg text-slate-500 font-medium mb-2">Số lượng peptide 50 amino acid có thể tồn tại:</p>
              <h4 className="text-5xl font-black text-rose-600 dark:text-rose-500 drop-shadow-sm flex justify-center items-center gap-3">
                <span className="text-3xl text-slate-400">≈</span> 20<sup>50</sup>
              </h4>
            </div>
            <p className="font-medium text-slate-700 dark:text-slate-300 leading-relaxed text-center">
              Lớn hơn số lượng nguyên tử trong vũ trụ. Không thể rà soát thực nghiệm mà <strong>bắt buộc phải cần đến Học sâu tạo sinh.</strong>
            </p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 3,
    title: "2. Các hướng mô hình nổi bật trong thiết kế peptide",
    content: (
      <div className="flex flex-col h-full space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 flex-1">
          {/* RNN */}
          <div className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div>
              <div className="w-14 h-14 rounded-xl bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center mb-4 text-orange-600">
                <span className="material-symbols-outlined text-3xl">psychology</span>
              </div>
              <h4 className="font-bold text-lg text-slate-800 dark:text-slate-100 mb-2">RNN / LSTM</h4>
              <p className="text-sm font-medium text-slate-600 dark:text-slate-400">Sinh chuỗi tuần tự theo từng Token (chữ cái).</p>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-600 text-sm">
              <div className="text-emerald-600 dark:text-emerald-400 font-semibold mb-1">+ Rất nhẹ, phản xạ nhanh</div>
              <div className="text-rose-600 dark:text-rose-400 font-semibold">- Mất ngữ cảnh, mù cấu trúc 3D</div>
            </div>
          </div>
          {/* Transformer */}
          <div className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div>
              <div className="w-14 h-14 rounded-xl bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mb-4 text-blue-600">
                <span className="material-symbols-outlined text-3xl">hub</span>
              </div>
              <h4 className="font-bold text-lg text-slate-800 dark:text-slate-100 mb-2">Transformer (PLMs)</h4>
              <p className="text-sm font-medium text-slate-600 dark:text-slate-400">Học ngôn ngữ Protein (VD: ESM-2, AlphaFold2).</p>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-600 text-sm">
              <div className="text-emerald-600 dark:text-emerald-400 font-semibold mb-1">+ Tri thức tiến hóa cực sâu</div>
              <div className="text-rose-600 dark:text-rose-400 font-semibold">- Hàng Tỷ tham số, chi phí VRAM KHỔNG LỒ</div>
            </div>
          </div>
          {/* GNN */}
          <div className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div>
              <div className="w-14 h-14 rounded-xl bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center mb-4 text-emerald-600">
                <span className="material-symbols-outlined text-3xl">scatter_plot</span>
              </div>
              <h4 className="font-bold text-lg text-slate-800 dark:text-slate-100 mb-2">Graph Neural Networks</h4>
              <p className="text-sm font-medium text-slate-600 dark:text-slate-400">Mô hình hóa tương tác đỉnh (Node) và cạnh giới hạn Không gian 3D.</p>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-600 text-sm">
              <div className="text-emerald-600 dark:text-emerald-400 font-semibold mb-1">+ Bắt được bản đồ gấp cuộn Local</div>
              <div className="text-rose-600 dark:text-rose-400 font-semibold">- Chỉ xếp xỉ liên kết tĩnh bề mặt</div>
            </div>
          </div>
          {/* GAN */}
          <div className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col justify-between hover:shadow-md transition">
            <div>
              <div className="w-14 h-14 rounded-xl bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center mb-4 text-purple-600">
                <span className="material-symbols-outlined text-3xl">balance</span>
              </div>
              <h4 className="font-bold text-lg text-slate-800 dark:text-slate-100 mb-2">GAN Generator</h4>
              <p className="text-sm font-medium text-slate-600 dark:text-slate-400">Bộ tạo (Sinh) đấu đối kháng với Bộ phân loại (Discriminator).</p>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-600 text-sm">
              <div className="text-emerald-600 dark:text-emerald-400 font-semibold mb-1">+ Tăng Realism của Data sinh ra</div>
              <div className="text-rose-600 dark:text-rose-400 font-semibold">- Dễ đứt gãy Gradient do "Mode Collapse" rời rạc</div>
            </div>
          </div>
        </div>
        <div className="mt-4 bg-indigo-50 dark:bg-indigo-900/10 border-l-4 border-indigo-500 p-6 rounded-r-xl shadow-inner font-medium text-xl text-indigo-900 dark:text-indigo-200">
          <strong>👉 Insight quan trọng:</strong> KHÔNG CÓ một kiến trúc nào đơn lẻ tự giải quyết hoàn hảo bài toán. Đây là tiền đề cho giải pháp **Kiến trúc Lai (Hybrid)** ra đời.
        </div>
      </div>
    )
  },
  {
    id: 4,
    title: "3. Triết lý thiết kế kiến trúc LightweightPeptideGen",
    content: (
      <div className="grid md:grid-cols-2 gap-8 h-full items-center">
        <div className="bg-slate-800 text-slate-100 p-8 rounded-3xl shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full blur-2xl"></div>
          <h3 className="text-3xl font-black mb-8 border-b border-slate-600 pb-4">Tứ Trụ Kiến Trúc</h3>
          <ul className="space-y-6">
            <li className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-500/20 text-blue-400 rounded-xl flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined">auto_stories</span>
              </div>
              <div>
                <strong className="block text-xl">1. Giữ tri thức Tự nhiên (Evolution)</strong>
                <span className="text-sm text-slate-400">KHÔNG Train chuỗi từ đầu. Kế thừa ngôn ngữ Protein lõi.</span>
              </div>
            </li>
            <li className="flex items-center gap-4">
              <div className="w-12 h-12 bg-emerald-500/20 text-emerald-400 rounded-xl flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined">shape_line</span>
              </div>
              <div>
                <strong className="block text-xl">2. Ép hiểu Hình học (Spatial Layout)</strong>
                <span className="text-sm text-slate-400">Không bỏ qua tương tác khoảng cách giữa các Nodes Cα.</span>
              </div>
            </li>
            <li className="flex items-center gap-4">
              <div className="w-12 h-12 bg-amber-500/20 text-amber-400 rounded-xl flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined">science</span>
              </div>
              <div>
                <strong className="block text-xl">3. Kiểm soát Vật lý hóa (Biophysics)</strong>
                <span className="text-sm text-slate-400">Giới hạn đặc tính tĩnh điện, ưa nước để kháng "Ảo giác sinh học".</span>
              </div>
            </li>
            <li className="flex items-center gap-4">
              <div className="w-12 h-12 bg-rose-500/20 text-rose-400 rounded-xl flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined">bolt</span>
              </div>
              <div>
                <strong className="block text-xl">4. Tối Ưu Tài Nguyên (Lightweight)</strong>
                <span className="text-sm text-slate-400">Toàn bộ chạy được trên Local Desktop GPU truyền thống (&lt;24GB).</span>
              </div>
            </li>
          </ul>
        </div>
        <div className="text-center px-8">
          <span className="material-symbols-outlined text-[8rem] text-slate-200 dark:text-slate-800 drop-shadow mb-6">extension</span>
          <p className="text-2xl font-medium leading-relaxed text-slate-700 dark:text-slate-300">
            Mục tiêu là xây dựng khối <strong>Generator Thông Minh</strong> vừa có độ sáng tạo từ tính Ngẫu nhiên của Text, vừa bó hẹp hợp lý bởi tính Logic của Hóa Học.
          </p>
        </div>
      </div>
    )
  },
  {
    id: 5,
    title: "4. Tổng quan kiến trúc mô hình",
    content: (
      <div className="flex flex-col items-center justify-center gap-6 h-full w-full max-w-6xl mx-auto">
        {/* Flow Map UI (Top to Bottom visual) */}
        <div className="w-full bg-white/60 dark:bg-slate-900/60 rounded-3xl p-6 border border-slate-200 dark:border-slate-700 shadow-sm relative">
          <div className="flex flex-col md:flex-row justify-between items-center text-center font-bold text-[0.8rem] md:text-sm lg:text-base mb-6 text-slate-500 gap-2">
            <div className="bg-slate-100 dark:bg-slate-800 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-700 grow">Input Peptide</div>
            <span className="material-symbols-outlined text-xl rotate-90 md:rotate-0">arrow_forward</span>
            <div className="bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 px-4 py-2 rounded-xl border border-blue-200 dark:border-blue-700 grow">ESM-2 Encoding</div>
            <span className="material-symbols-outlined text-xl rotate-90 md:rotate-0">arrow_forward</span>
            <div className="bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 px-4 py-2 rounded-xl border border-purple-200 dark:border-purple-700 grow">GATv2 Graph</div>
            <span className="material-symbols-outlined text-xl rotate-90 md:rotate-0">arrow_forward</span>
            <div className="bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 px-4 py-2 rounded-xl border border-amber-200 dark:border-amber-700 grow">BioPhysical Fusion</div>
            <span className="material-symbols-outlined text-xl rotate-90 md:rotate-0">arrow_forward</span>
            <div className="bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300 px-4 py-2 rounded-xl border border-rose-200 dark:border-rose-700 grow">GRU (GAN)</div>
            <span className="material-symbols-outlined text-xl rotate-90 md:rotate-0">arrow_forward</span>
            <div className="bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 px-4 py-2 rounded-xl border border-emerald-200 dark:border-emerald-700 grow">RL Optimizer</div>
          </div>
          <div className="relative group w-full mx-auto">
            <img
              src="/Peptide-Design-Lightweight-Model/ESM2-GAT.png"
              alt="Architecture Overview"
              className="relative w-full rounded-2xl shadow object-contain bg-white dark:bg-slate-800 max-h-[450px]"
            />
          </div>
        </div>
      </div>
    )
  },
  {
    id: 6,
    title: "5. Thành phần 1 — Sequence Backbone (ESM-2 Lightweight)",
    content: (
      <div className="grid md:grid-cols-2 gap-10 items-center h-full">
        <div className="flex justify-center h-full">
          <img src="/Peptide-Design-Lightweight-Model/esm2.png" alt="ESM-2 Architecture" className="rounded-3xl border border-slate-200 dark:border-slate-700 shadow-lg object-cover bg-white w-full h-full max-h-[480px]" />
        </div>
        <div className="space-y-6 text-xl">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-8 rounded-3xl border border-blue-200 dark:border-blue-800 shadow-sm">
             <p className="font-semibold text-blue-900 dark:text-blue-100 leading-relaxed mb-6">
              ESM-2 (Evolutionary Scale Modeling) đóng vai trò là cốt lõi <strong className="text-blue-600 dark:text-blue-400">Mã hóa Tri thức tiến hóa Sequences</strong>.
            </p>
            <h4 className="font-black text-2xl text-blue-700 dark:text-blue-300 mb-4 border-b border-blue-200 dark:border-blue-800 pb-2">Chiến lược: Selective Freezing</h4>
            <ul className="list-disc pl-6 space-y-4 text-slate-700 dark:text-slate-300 font-medium">
              <li>
                <strong>Đóng băng (Freeze) 90-95% Layers đầu:</strong> Giữ nguyên "semantic tiến hóa" học được từ phân tích 250 triệu chuỗi Protein thuộc cơ sở dữ liệu UniRef. Giúp tiết kiệm cực kỳ lớn tính toán Gradient.
              </li>
              <li>
                <strong>Fine-tune 2 lơp Transformer cuối:</strong> Ép mô hình chuyển sự tập trung vào "ngữ pháp sinh học" đặc thù của <em>Peptide ngắn kháng khuẩn</em> thay vì các phân tử khổng lồ.
              </li>
            </ul>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 7,
    title: "6. Thành phần 2 — Spatial Graph Encoder (GATv2)",
    content: (
      <div className="grid md:grid-cols-2 gap-10 items-center h-full">
        <div className="space-y-6 text-xl order-2 md:order-1">
          <div className="bg-purple-50 dark:bg-purple-900/20 p-8 rounded-3xl border border-purple-200 dark:border-purple-800 shadow-sm">
            <p className="font-semibold text-purple-900 dark:text-purple-100 leading-relaxed mb-6">
              Chuyển đổi chuỗi 1D thành <strong>Đồ thị Không gian 3D</strong> thông qua Mạng Graph Attention Networks (GATv2).
            </p>
            <ul className="list-disc pl-6 space-y-4 text-slate-700 dark:text-slate-300 font-medium mb-6">
              <li><strong>Node (Đỉnh):</strong> Amino acid mang embedding 20 chiều đặc thù chữ cái.</li>
              <li><strong>Edge (Cạnh):</strong> Tương tác vật lý bán kính <code>&lt; 8Å</code> quét dọc xương sống phân tử Cα. Bắt sóng tương tác tĩnh điện / Van der Waals / H-Bond.</li>
            </ul>
            <div className="bg-white/60 dark:bg-slate-900/60 p-4 rounded-xl border border-purple-200/50 dark:border-purple-800 text-base italic text-slate-600 dark:text-slate-400">
              <strong className="text-purple-600 dark:text-purple-400">Vì sao GATv2?</strong> Khác GATv1 bị kẹt "Static attention", GATv2 sở hữu cơ chế Attention Động kết hợp 4 Heads tăng cường phân giải sức hút (rank) giữa Node X và Y liên tục biến hóa chiều sâu. Mô hình **vừa hiểu thứ tự, vừa hiểu cách gấp cuộn (Folding).**
            </div>
          </div>
        </div>
        <div className="flex justify-center h-full order-1 md:order-2">
          <img src="/Peptide-Design-Lightweight-Model/gatarchi.png" alt="GATv2 Architecture" className="rounded-3xl border border-slate-200 dark:border-slate-700 shadow-lg object-cover bg-white p-4 w-full h-full max-h-[480px]" />
        </div>
      </div>
    )
  },
  {
    id: 8,
    title: "7. Thành phần 3 — Biophysical Conditional Control",
    content: (
      <div className="space-y-6 h-full flex flex-col justify-center">
        <div className="text-center max-w-4xl mx-auto mb-6">
          <h3 className="text-2xl font-black text-amber-600 dark:text-amber-400 mb-2">Đổi mới Đột phá: Sự tham gia của tham số Hóa Lý </h3>
          <p className="text-lg font-medium text-slate-600 dark:text-slate-300">
            Hạn chế "Ảo giác cấu trúc". Biến latent space tự do thành Không gian có <strong>điều kiện Giới hạn Vật lý</strong>. 
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-8 text-base">
          <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm hover:-translate-y-1 transition duration-300">
            <h4 className="font-bold border-b border-amber-200 dark:border-amber-900 pb-2 mb-4 text-amber-600 dark:text-amber-500">Stability & Safety</h4>
            <ul className="space-y-2 text-slate-600 dark:text-slate-400 font-medium">
              <li>• Instability Index (II)</li>
              <li>• Hemolytic Score (Toxicity)</li>
              <li>• Thermodynamic Stability</li>
              <li>• Boman Index</li>
              <li>• Aliphatic Index</li>
              <li>• Molecular Weight</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm hover:-translate-y-1 transition duration-300">
            <h4 className="font-bold border-b border-amber-200 dark:border-amber-900 pb-2 mb-4 text-amber-600 dark:text-amber-500">Hydrophobicity (Kỵ Nước)</h4>
            <ul className="space-y-2 text-slate-600 dark:text-slate-400 font-medium">
              <li>• GRAVY Score</li>
              <li>• Hydrophobic Moment</li>
              <li>• Solvation Free Energy</li>
              <li>• Amphiphilicity Score</li>
              <li>• Hydrophobicity Profile</li>
              <li>• Sequence Entropy</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm hover:-translate-y-1 transition duration-300">
            <h4 className="font-bold border-b border-amber-200 dark:border-amber-900 pb-2 mb-4 text-amber-600 dark:text-amber-500">Charge & Structure</h4>
            <ul className="space-y-2 text-slate-600 dark:text-slate-400 font-medium">
              <li>• Total Net Charge (pH 7)</li>
              <li>• Isoelectric Point (pI)</li>
              <li>• Aromaticity</li>
              <li>• Polarity Level</li>
              <li>• Flexibility Index</li>
              <li>• Helix / Sheet Ratio</li>
            </ul>
          </div>
        </div>

        <div className="mx-auto w-full max-w-4xl bg-amber-50 dark:bg-amber-900/10 rounded-2xl p-4 border border-amber-200 dark:border-amber-800 flex items-center justify-center gap-4 text-lg font-bold text-amber-800 dark:text-amber-200">
          <span className="material-symbols-outlined text-3xl">compress</span>
          <span>(18 Features Vector &rarr; Chuẩn hóa Z-Score) &times; W(Linear Projection) &rarr; <span className="underline">Đưa vào Cross-Attention Cơ chế</span></span>
        </div>
      </div>
    )
  },
  {
    id: 9,
    title: "8. Thành phần 4 — Generative Core (GRU-GAN)",
    content: (
      <div className="grid md:grid-cols-2 gap-8 h-full">
        <div className="bg-rose-50 dark:bg-rose-900/10 p-8 rounded-3xl border border-rose-200 dark:border-rose-900 shadow-sm relative overflow-hidden flex flex-col justify-center">
          <span className="material-symbols-outlined absolute right-[-20px] top-[-20px] text-[150px] text-rose-500/10">edit</span>
          <h3 className="text-3xl font-black text-rose-600 dark:text-rose-400 mb-6">Bộ Sinh (Generator)</h3>
          <ul className="space-y-5 text-xl font-medium text-slate-700 dark:text-slate-300">
            <li><strong>Lõi GRU:</strong> Thay vì Decoder Transformer khổng lồ, GRU được giữ lại để sinh Text Token (Axit Amin) qua từng Time-step. Nhẹ, nhanh.</li>
            <li><strong>Gumbel-Softmax Tricking:</strong> Mô hình sinh ra Token rời rạc (One-hot A,T,G,C...). Để Gradient được phép chảy ngược khi huấn luyện GAN, sử dụng độ cong Gumbel.</li>
            <li><strong>Nucleus Sampling (Top-P 0.9):</strong> Cắt ngọn xác suất kỳ lạ, tăng tỉ lệ sinh đa dạng.</li>
          </ul>
        </div>

        <div className="bg-slate-100 dark:bg-slate-800 p-8 rounded-3xl border border-slate-300 dark:border-slate-700 shadow-sm relative overflow-hidden flex flex-col justify-center">
          <span className="material-symbols-outlined absolute right-[-20px] top-[-20px] text-[150px] text-slate-500/10">radar</span>
          <h3 className="text-3xl font-black text-slate-800 dark:text-slate-100 mb-6">Bộ Giám Khảo (Discriminator)</h3>
          <ul className="space-y-5 text-xl font-medium text-slate-600 dark:text-slate-400">
            <li><strong>Lớp CNN 1D:</strong> Quét dọc qua toàn bộ chuỗi được sinh ra để nhìn Local Pattern (VD: Cụm Kỵ nước liên tiếp).</li>
            <li><strong>Wasserstein GAN (WGAN):</strong> Sử dụng Earth Mover Distance đo khoảng cách phân phối Giả và Thật mượt mà hơn Jensen-Shannon chuẩn.</li>
            <li><strong>Gradient Penalty:</strong> Trừng phạt Loss giúp GAN không bị suy sụp hàm toán. Tăng ranh giới chất lượng realism tối đa lên Peptide do AI sinh ra.</li>
          </ul>
        </div>
      </div>
    )
  },
  {
    id: 10,
    title: "9. Thành phần 5 — Reinforcement Learning Optimization",
    content: (
      <div className="flex flex-col h-full bg-emerald-50 dark:bg-emerald-900/10 rounded-3xl border border-emerald-200 dark:border-emerald-800 p-8 shadow-sm">
        <h3 className="text-2xl font-black text-emerald-700 dark:text-emerald-400 mb-6 border-b border-emerald-200 dark:border-emerald-800 pb-3 flex items-center gap-2">
          <span className="material-symbols-outlined text-3xl">sports_esports</span> Giai đoạn Học Tăng Cường (RL)
        </h3>
        <p className="text-xl font-medium text-slate-700 dark:text-slate-300 mb-8 leading-relaxed">
          Sử dụng kỹ thuật <strong className="text-emerald-600 dark:text-emerald-400">SCST (Self-Critical Sequence Training) - thuộc dạng Policy Gradient.</strong>
          <br/>
          Mô hình không chỉ sinh chuỗi y hệt Data cũ, mà được thưởng lớn khi Peptide do AI sáng tác <strong>đạt các chuẩn mực Y khoa.</strong>
        </p>
        
        <div className="grid md:grid-cols-2 gap-8 text-lg flex-1">
          <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-emerald-100 dark:border-emerald-900 flex flex-col justify-center shadow">
             <h4 className="font-bold text-emerald-600 mb-4 text-xl">Thuật toán Loss (Policy)</h4>
             <p className="text-slate-600 dark:text-slate-400 italic mb-4">RL ép sinh Sample, tính Loss, và so sánh với Baseline trung bình nội tại để ban Reward tích cực/tiêu cực.</p>
             <code className="bg-slate-100 dark:bg-slate-900 p-4 rounded text-sm text-pink-600 dark:text-pink-400 mx-auto w-full inline-block font-mono">
                Loss_RL = - (Reward_sample - Reward_baseline) * log(Prob_sample)
             </code>
          </div>
          <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-emerald-100 dark:border-emerald-900 flex flex-col justify-center shadow">
             <h4 className="font-bold text-emerald-600 mb-4 text-xl">Reward Function kết hợp:</h4>
             <ul className="space-y-3 font-semibold text-slate-700 dark:text-slate-300">
               <li className="flex items-center gap-2"><span className="text-xl">👉</span> R_Stability: Thưởng khi Instability Index &lt; 40.</li>
               <li className="flex items-center gap-2"><span className="text-xl">👉</span> R_Tox: Phạt cực mạnh rủi ro khi Hemolytic cao.</li>
               <li className="flex items-center gap-2"><span className="text-xl">👉</span> R_Hydrophobic: Trừu tượng hóa lực GRAVY.</li>
               <li className="flex items-center gap-2"><span className="text-xl">👉</span> R_Thermo: Proxy năng lượng liên kết động.</li>
             </ul>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 11,
    title: "10. Logic huấn luyện 3 pha",
    content: (
      <div className="space-y-6 text-xl h-full flex flex-col justify-center">
        <div className="grid grid-cols-1 gap-6 relative max-w-5xl mx-auto w-full">
          {/* Vertical line through timeline */}
          <div className="absolute left-[3.25rem] top-10 bottom-10 w-1 bg-slate-200 dark:bg-slate-700 z-0 hidden md:block"></div>

          <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-md border border-slate-200 dark:border-slate-700 flex flex-col md:flex-row gap-6 relative z-10 transition hover:-translate-y-1">
            <div className="bg-blue-100 dark:bg-blue-900/80 text-blue-600 dark:text-blue-400 font-black text-3xl h-16 w-16 rounded-full flex items-center justify-center shrink-0 border-[6px] border-white dark:border-slate-900 shadow-sm mt-1">1</div>
            <div>
              <h3 className="font-black text-2xl text-blue-600 dark:text-blue-400 mb-2 uppercase tracking-wide">Phase 1 — Distribution Learning</h3>
              <p className="text-lg text-slate-600 dark:text-slate-300 font-medium leading-relaxed">Mô hình sơ cấp học "Ngữ pháp Peptide" qua phân phối dữ liệu chuẩn bằng <strong>Maximum Likelihood Estimation (MLE) / Teacher Forcing (CrossEntropy)</strong>.</p>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-md border border-slate-200 dark:border-slate-700 flex flex-col md:flex-row gap-6 relative z-10 transition hover:-translate-y-1">
            <div className="bg-purple-100 dark:bg-purple-900/80 text-purple-600 dark:text-purple-400 font-black text-3xl h-16 w-16 rounded-full flex items-center justify-center shrink-0 border-[6px] border-white dark:border-slate-900 shadow-sm mt-1">2</div>
            <div>
              <h3 className="font-black text-2xl text-purple-600 dark:text-purple-400 mb-2 uppercase tracking-wide">Phase 2 — Realism Learning</h3>
              <p className="text-lg text-slate-600 dark:text-slate-300 font-medium leading-relaxed">GAN bước vào đấu trường. Học "Tính thực tế, có nghĩa tự nhiên". Trừng phạt chuỗi nhảm nhí thông qua <strong>Adversarial Loss từ CNN WGAN-GP</strong>.</p>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-md border border-slate-200 dark:border-slate-700 flex flex-col md:flex-row gap-6 relative z-10 transition hover:-translate-y-1">
            <div className="bg-emerald-100 dark:bg-emerald-900/80 text-emerald-600 dark:text-emerald-400 font-black text-3xl h-16 w-16 rounded-full flex items-center justify-center shrink-0 border-[6px] border-white dark:border-slate-900 shadow-sm mt-1">3</div>
            <div>
              <h3 className="font-black text-2xl text-emerald-600 dark:text-emerald-400 mb-2 uppercase tracking-wide">Phase 3 — Objective Learning</h3>
              <p className="text-lg text-slate-600 dark:text-slate-300 font-medium leading-relaxed">Tinh hoa sau cùng. <strong>Reinforcement Learning (SCST)</strong> kéo phân phối đồ thị chuỗi về vị trí "Tối ưu Sinh học" ép tiêu chuẩn màng kháng khuẩn chuẩn xác nhất.</p>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 12,
    title: "11. Kết quả kiến trúc mang lại",
    content: (
      <div className="space-y-6 flex flex-col justify-center h-full">
        <ul className="flex flex-wrap gap-4 text-base font-bold text-slate-700 dark:text-slate-200 mb-6 bg-slate-100 dark:bg-slate-800 p-6 rounded-2xl shadow-inner border border-slate-200 dark:border-slate-700 justify-center">
          <li className="flex items-center gap-2"><span className="material-symbols-outlined text-green-500">check_circle</span> Diversity Score ~0.65 vượt trội mọi Baseline</li>
          <li className="flex items-center gap-2"><span className="material-symbols-outlined text-green-500">check_circle</span> Sinh học In-silico chuẩn xác</li>
          <li className="flex items-center gap-2"><span className="material-symbols-outlined text-blue-500">speed</span> Sinh 1000 peptide &lt; 2s</li>
          <li className="flex items-center gap-2"><span className="material-symbols-outlined text-purple-500">memory</span> VRAM Inference thu gọn ~12GB (Rất nhẹ)</li>
        </ul>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-4 bg-white/50 dark:bg-slate-800/50 rounded-2xl border border-slate-200 dark:border-slate-700 text-center">
            <h4 className="font-bold text-emerald-600 dark:text-emerald-400 mb-2">Độ ổn định (Instability Index)</h4>
            <img src="/Peptide-Design-Lightweight-Model/instability_hist.png" alt="Instability Index" className="w-full h-auto rounded-lg mb-2 border border-slate-200 dark:border-slate-600 aspect-video object-cover" />
            <p className="text-sm font-medium">95% tập trung vùng &lt;40 Threshold siêu bền cấu trúc.</p>
          </div>
          <div className="p-4 bg-white/50 dark:bg-slate-800/50 rounded-2xl border border-slate-200 dark:border-slate-700 text-center">
            <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-2">Phân bố Độc tính (Hemolytic)</h4>
            <img src="/Peptide-Design-Lightweight-Model/hemolytic_score_hist.png" alt="Hemolytic" className="w-full h-auto rounded-lg mb-2 border border-slate-200 dark:border-slate-600 aspect-video object-cover" />
            <p className="text-sm font-medium">Đồ thị hình chuông đẩy gọn về dải điểm an toàn tiêu diệt Bacteria.</p>
          </div>
          <div className="p-4 bg-white/50 dark:bg-slate-800/50 rounded-2xl border border-slate-200 dark:border-slate-700 text-center">
            <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-2">Kỵ nước xuyên màng (GRAVY)</h4>
            <img src="/Peptide-Design-Lightweight-Model/gravy_hist.png" alt="GRAVY" className="w-full h-auto rounded-lg mb-2 border border-slate-200 dark:border-slate-600 aspect-video object-cover" />
            <p className="text-sm font-medium">Distribution dao động hoàn hảo trong khoảng [-1.0, 0.5] lý thuyết.</p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 13,
    title: "12. Đóng góp khoa học của kiến trúc",
    content: (
      <div className="bg-indigo-50 dark:bg-indigo-900/10 p-10 rounded-3xl border border-indigo-200 dark:border-indigo-900/50 shadow-sm h-full flex flex-col justify-center">
        <div className="text-center mb-10 pb-6 border-b-2 border-indigo-200 dark:border-indigo-800 max-w-3xl mx-auto">
          <h3 className="text-3xl font-black text-indigo-700 dark:text-indigo-400 leading-snug">Chuyển hướng Đột Phá — Thiết Kế Cấu Trúc Bỏ Qua "Brute-force Compute"</h3>
        </div>
        <div className="grid md:grid-cols-2 gap-8 text-xl font-medium text-slate-800 dark:text-slate-200">
          <div className="flex gap-4 p-5 bg-white/60 dark:bg-slate-800/60 rounded-2xl">
            <span className="material-symbols-outlined text-4xl text-indigo-500">hub</span>
            <div><strong className="block text-indigo-700 dark:text-indigo-300 mb-1">Fusion Đa Thang Đo Hoàn Mỹ</strong> 
             Lần đầu hợp nhất cả 3 chiều: Sequence (ESM-2) + Spatial (GATv2) + Hóa Lý (Z-Score) vào Generator Token.</div>
          </div>
          <div className="flex gap-4 p-5 bg-white/60 dark:bg-slate-800/60 rounded-2xl">
            <span className="material-symbols-outlined text-4xl text-emerald-500">route</span>
            <div><strong className="block text-emerald-700 dark:text-emerald-300 mb-1">Pipeline Training Xuyên Chéo Chuỗi</strong> 
             Giải quyết mượt mà đứt gãy không gian rời rạc qua quy trình Loss 3 pha MLE &rarr; GAN (Gumbel-Softmax) &rarr; SCST Reward.</div>
          </div>
          <div className="flex gap-4 p-5 bg-white/60 dark:bg-slate-800/60 rounded-2xl">
            <span className="material-symbols-outlined text-4xl text-blue-500">memory</span>
            <div><strong className="block text-blue-700 dark:text-blue-300 mb-1">Resource-Efficient Design</strong> 
             Nâng cao hiệu suất bằng hàng loạt Trick: Layer Freeze Backbone, GAT Attention Cα động. Ít thông số nhưng Sức mạnh cực đại.</div>
          </div>
          <div className="flex gap-4 p-5 bg-white/60 dark:bg-slate-800/60 rounded-2xl">
            <span className="material-symbols-outlined text-4xl text-rose-500">public</span>
            <div><strong className="block text-rose-700 dark:text-rose-300 mb-1">Dân Chủ Hóa (Democratization) AI Lab</strong> 
             Mở ra khả năng nghiên cứu in-silico tốc độ siêu bão cho các tổ chức khoa học, môi trường ĐH Y-Dược hạn chế tài nguyên Khủng ở VN.</div>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 14,
    title: "13. Hạn chế kiến trúc",
    content: (
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 h-full items-center">
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[280px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">view_in_ar</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">GATv2 tĩnh & Thiếu Động Lực Học</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Đồ thị chỉ xấp xỉ liên kết tĩnh, không mô phỏng biến thiên quỹ đạo thời gian thật - Molecular Dynamics Simulation.</p>
        </div>
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[280px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">function</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Reward vẫn là Proxy</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Các phần thưởng Instability hay GRAVY là suy diễn toán học in-silico, chưa gắn với chỉ số lâm sàng MIC hay IC50 trực tiếp.</p>
        </div>
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[280px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">layers</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Thiếu Folding Runtime</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Module tạo sinh chạy chưa nhúng thuật toán PDB 3D Fold check trực tiếp vào trong Network Loss trong lúc Train.</p>
        </div>
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[280px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">database</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Data Bias Base-Models</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Quá mạnh trên khu vực AMP Data của DBAASP, nhưng có thể sinh lạc lối đối với dữ liệu Protein miễn dịch đặc dạng khác.</p>
        </div>
      </div>
    )
  },
  {
    id: 15,
    title: "14. Hướng phát triển kiến trúc",
    content: (
      <div className="flex flex-col justify-center h-full">
        <h3 className="text-2xl font-black text-emerald-600 dark:text-emerald-400 mb-8 max-w-4xl text-center mx-auto uppercase tracking-wide">Tương Lai Tiến Trình Thiết Kế Ai Peptide</h3>
        <div className="max-w-4xl mx-auto w-full space-y-4">
          <div className="bg-white/60 dark:bg-slate-800/60 p-5 rounded-2xl border border-slate-200 dark:border-slate-700 flex gap-4 items-center">
             <div className="bg-emerald-500 p-2 rounded-xl text-white flex shrink-0"><span className="material-symbols-outlined font-bold">account_tree</span></div>
             <p className="text-lg font-bold text-slate-700 dark:text-slate-200">Tích hợp AlphaFold3 / ESMFold Runtime làm Reward Function thời gian thực.</p>
          </div>
          <div className="bg-white/60 dark:bg-slate-800/60 p-5 rounded-2xl border border-slate-200 dark:border-slate-700 flex gap-4 items-center">
             <div className="bg-emerald-500 p-2 rounded-xl text-white flex shrink-0"><span className="material-symbols-outlined font-bold">rocket_launch</span></div>
             <p className="text-lg font-bold text-slate-700 dark:text-slate-200">Pareto RL đa mục tiêu: Sinh chuỗi cực bền đồng thời Cực ít độc tố cho mô người.</p>
          </div>
          <div className="bg-white/60 dark:bg-slate-800/60 p-5 rounded-2xl border border-slate-200 dark:border-slate-700 flex gap-4 items-center">
             <div className="bg-emerald-500 p-2 rounded-xl text-white flex shrink-0"><span className="material-symbols-outlined font-bold">water_drop</span></div>
             <p className="text-lg font-bold text-slate-700 dark:text-slate-200">Diffusion Model Layer: Ứng dụng mô hình khuếch tán để Refine Noise Cα Structure.</p>
          </div>
          <div className="bg-white/60 dark:bg-slate-800/60 p-5 rounded-2xl border border-slate-200 dark:border-slate-700 flex gap-4 items-center">
             <div className="bg-emerald-500 p-2 rounded-xl text-white flex shrink-0"><span className="material-symbols-outlined font-bold">science</span></div>
             <p className="text-lg font-bold text-slate-700 dark:text-slate-200">Wet-lab Verification: Feedback loop giữa AI & Máy thử nghiệm lâm sàng thực tiễn sinh học ướt.</p>
          </div>
          <div className="bg-white/60 dark:bg-slate-800/60 p-5 rounded-2xl border border-slate-200 dark:border-slate-700 flex gap-4 items-center shadow-md">
             <div className="bg-blue-600 p-2 rounded-xl text-white flex shrink-0"><span className="material-symbols-outlined font-bold">language</span></div>
             <p className="text-lg font-bold text-blue-800 dark:text-blue-300">Hoàn thiện mô hình Web SaaS (Software as a Service) phục vụ toàn cầu.</p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 16,
    title: "",
    content: (
      <div className="flex flex-col items-center justify-center h-full text-center space-y-10">
        <h2 className="text-5xl md:text-[4rem] font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 via-blue-500 to-purple-600 pb-2 mb-2">15. DEMO SẢN PHẨM</h2>
        <p className="text-2xl font-medium text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Cùng trải nghiệm khả năng sinh tạo chuỗi Peptide siêu tốc với độ tin cậy kết quả Hóa-Lý đo được trực tiếp ngay trên Giao diện Web.
        </p>
        <a 
          href="/Peptide-Design-Lightweight-Model/generation"
          className="group relative px-12 py-5 font-bold text-white rounded-full bg-slate-900 dark:bg-white dark:text-slate-900 text-2xl overflow-hidden hover:scale-105 transition-transform duration-300 shadow-2xl flex items-center justify-center gap-3"
        >
          <div className="absolute inset-0 w-full h-full border-[6px] border-emerald-500/30 rounded-full blur-[10px] scale-110 group-hover:blur-[20px] transition-all"></div>
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-emerald-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <span className="relative z-10 flex items-center gap-3">
             Tiến hành Thử nghiệm 
             <span className="material-symbols-outlined text-3xl transition-transform group-hover:translate-x-2">arrow_forward</span>
          </span>
        </a>
      </div>
    )
  }
];

const Documentation = () => {
  const containerRef = useRef(null);

  const handlePresentationClick = (e) => {
    // Prevent triggering on interactive elements or text selection
    if (e.target.tagName.toLowerCase() === 'a' || e.target.closest('a')) return;
    if (e.target.tagName.toLowerCase() === 'button' || e.target.closest('button')) return;
    if (window.getSelection().toString().length > 0) return;

    if (!containerRef.current) return;
    const sections = Array.from(containerRef.current.querySelectorAll('section'));
    if (sections.length === 0) return;

    // Calculate the horizontal middle of the viewport
    const viewportMiddle = window.scrollY + (window.innerHeight / 2);

    // Find the first section whose top edge is below the viewport middle
    let nextSection = null;
    for (const section of sections) {
      if (section.offsetTop > viewportMiddle + 50) { // add small buffer
        nextSection = section;
        break;
      }
    }

    if (nextSection) {
      nextSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  return (
    <div className="flex bg-slate-50 dark:bg-background-dark text-slate-800 dark:text-slate-200 min-h-screen overflow-x-hidden">
      <main
        className="flex-1 w-full scroll-smooth flex flex-col items-center overflow-x-hidden min-h-screen"
        ref={containerRef}
        onClick={handlePresentationClick}
      >

        <div className="pb-32 w-full flex flex-col items-center">
          {slidesData.map((slide) => (
            <section
              key={slide.id}
              id={`slide-${slide.id}`}
              className={`${slidestyle} ${slide.important ? 'border-amber-400 dark:border-amber-500 shadow-amber-200/50 dark:shadow-amber-900/30 ring-4 ring-amber-400/20' : ''}`}
            >
              <div className={`absolute top-0 right-0 w-[500px] h-[500px] rounded-full blur-[100px] -z-10 transition-all duration-700 ${slide.id % 3 === 0 ? 'bg-blue-500/10 group-hover:bg-blue-500/20' : slide.id % 3 === 1 ? 'bg-purple-500/10 group-hover:bg-purple-500/20' : 'bg-emerald-500/10 group-hover:bg-emerald-500/20'}`}></div>
              <div className={`absolute bottom-0 left-0 w-[400px] h-[400px] rounded-full blur-[80px] -z-10 transition-all duration-700 ${slide.id % 3 === 0 ? 'bg-emerald-500/5' : slide.id % 3 === 1 ? 'bg-blue-500/5' : 'bg-purple-500/5'}`}></div>
              
              {/* VLU LOGO FIX */}
              <img
                src="/Peptide-Design-Lightweight-Model/logovlu.png"
                alt="Logo VLU"
                className="absolute top-[25px] right-5 h-10 md:h-12 object-contain opacity-80 z-20 pointer-events-none"
              />

              <div className="relative z-10 flex flex-col h-full w-full">
                {slide.title && (
                  <div className="mb-10 text-center md:text-left shrink-0">
                    <h2 className="text-4xl lg:text-5xl font-black border-b-4 border-slate-300 dark:border-slate-600 inline-block pb-3 text-slate-800 dark:text-slate-100 drop-shadow-sm">
                      {slide.title}
                    </h2>
                  </div>
                )}
                <div className="flex-1 w-full max-w-none flex flex-col justify-center">
                  <div className="w-full h-full flex flex-col">
                    {slide.content}
                  </div>
                </div>
              </div>

            </section>
          ))}
        </div>

        <div className="text-center pt-16 border-t border-slate-200 dark:border-slate-700 opacity-60 relative z-10 mb-8 w-full max-w-[1900px] px-8">
          <p className="text-lg font-medium tracking-wide">Đồ án Tốt nghiệp — <strong className="text-primary font-bold">Quang Mỹ Tâm</strong></p>
        </div>
      </main>
    </div>
  );
};

export default Documentation;
