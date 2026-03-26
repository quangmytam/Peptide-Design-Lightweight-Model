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
    title: "1. Cơ sở sinh học - Peptide và Amino Acid",
    content: (
      <div className="grid md:grid-cols-2 gap-10 text-xl items-center h-full">
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-8 rounded-3xl border border-blue-200 dark:border-blue-800 shadow-sm relative">
            <h3 className="font-bold text-2xl text-blue-600 dark:text-blue-400 mb-4 flex items-center gap-2 border-b-2 border-blue-200 dark:border-blue-800 pb-3">
              <span className="material-symbols-outlined text-3xl">biotech</span> Peptide & Kháng sinh tự nhiên (AMPs)
            </h3>
            <p className="font-medium text-slate-700 dark:text-slate-300 leading-relaxed">
              <strong>Peptide</strong> là chuỗi ngắn từ 2–50 amino acid. <strong>AMPs</strong> (Antimicrobial Peptides) là "vũ khí tự nhiên" tiềm năng nhất để đối phó với tình trạng **Kháng kháng sinh (AMR)**.
            </p>
            <ul className="list-disc pl-6 mt-4 space-y-3 text-lg font-medium">
              <li>Amino acid là các "đơn vị cấu tạo sinh học" (20 loại tiêu chuẩn).</li>
              <li>AMPs tiêu diệt vi khuẩn bằng cách phá hủy màng tế bào.</li>
              <li>Thách thức: Đảm bảo chuỗi peptide vừa có hoạt tính mạnh, vừa ổn định về mặt cấu trúc.</li>
            </ul>
          </div>
        </div>
        <div className="space-y-6">
          <div className="bg-rose-50 dark:bg-rose-900/10 p-8 rounded-3xl border border-rose-200 dark:border-rose-800 shadow-sm">
            <h3 className="font-bold text-2xl text-rose-600 dark:text-rose-400 mb-4 flex items-center gap-2 border-b-2 border-rose-200 dark:border-rose-800 pb-3">
              <span className="material-symbols-outlined text-3xl">public</span> Không gian tổ hợp vô tận
            </h3>
            <div className="text-center my-6">
              <p className="text-lg text-slate-500 font-medium mb-2">Số lượng peptide dài 50 amino acid khả thi:</p>
              <h4 className="text-5xl font-black text-rose-600 dark:text-rose-500 drop-shadow-sm flex justify-center items-center gap-3">
                <span className="text-3xl text-slate-400">≈</span> 20<sup>50</sup>
              </h4>
            </div>
            <p className="font-medium text-slate-700 dark:text-slate-300 leading-relaxed text-center">
              Vượt xa số lượng nguyên tử trong vũ trụ. Việc tìm kiếm trong không gian này bắt buộc cần đến sức mạnh của <strong>Trí tuệ nhân tạo Tạo sinh (Generative AI).</strong>
            </p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 3,
    title: "2. Phát biểu bài toán và Động lực nghiên cứu",
    content: (
      <div className="grid md:grid-cols-2 gap-8 h-full items-center">
        <div className="space-y-6">
          <div className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm">
            <h4 className="font-black text-2xl text-slate-800 dark:text-slate-100 mb-4">Vấn đề của các mô hình hiện nay</h4>
            <ul className="space-y-4 text-lg">
              <li className="flex gap-3">
                <span className="text-rose-500 material-symbols-outlined shrink-0">error</span>
                <span><strong>Ảo giác cấu trúc (Hallucination):</strong> Sinh chuỗi đúng ngữ pháp nhưng không thể tồn tại ổn định trong 3D.</span>
              </li>
              <li className="flex gap-3">
                <span className="text-rose-500 material-symbols-outlined shrink-0">memory</span>
                <span><strong>Tài nguyên khổng lồ:</strong> Các mô hình như ESM-2 full fine-tuning đòi hỏi GPU đắt đỏ.</span>
              </li>
            </ul>
          </div>
        </div>
        <div className="bg-blue-600 text-white p-8 rounded-3xl shadow-xl relative overflow-hidden">
          <h3 className="text-3xl font-black mb-6">Mục tiêu của Đồ án</h3>
          <p className="text-xl leading-relaxed mb-6 font-medium italic">
            "Xây dựng một kiến trúc <strong>Lightweight (Nhẹ)</strong>, có khả năng chạy trên Local GPU truyền thống nhưng vẫn mang <strong>Sức mạnh cấu trúc (Structural-aware)</strong>."
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/10 p-4 rounded-xl">
              <span className="block text-2xl font-bold">II &lt; 40</span>
              <span className="text-xs">Độ ổn định tối ưu</span>
            </div>
            <div className="bg-white/10 p-4 rounded-xl">
              <span className="block text-2xl font-bold">&lt; 12GB</span>
              <span className="text-xs">VRAM Tối ưu hóa</span>
            </div>
          </div>
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
          <h3 className="text-3xl font-black mb-8 border-b border-slate-600 pb-4">Bốn Trụ Cột Kiến Trúc</h3>
          <ul className="space-y-6">
            <li className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-500/20 text-blue-400 rounded-xl flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined">auto_stories</span>
              </div>
              <div>
                <strong className="block text-xl">1. Kế thừa tri thức Tiến hóa</strong>
                <span className="text-sm text-slate-400">Không huấn luyện chuỗi từ đầu. Tận dụng ngôn ngữ Protein sẵn có.</span>
              </div>
            </li>
            <li className="flex items-center gap-4">
              <div className="w-12 h-12 bg-emerald-500/20 text-emerald-400 rounded-xl flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined">shape_line</span>
              </div>
              <div>
                <strong className="block text-xl">2. Am hiểu Cấu trúc Hình học</strong>
                <span className="text-sm text-slate-400">Tận dụng tương tác không gian giữa các nguyên tử Cα.</span>
              </div>
            </li>
            <li className="flex items-center gap-4">
              <div className="w-12 h-12 bg-amber-500/20 text-amber-400 rounded-xl flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined">science</span>
              </div>
              <div>
                <strong className="block text-xl">3. Kiểm soát Đặc tính Hóa-Lý</strong>
                <span className="text-sm text-slate-400">Ràng buộc về tĩnh điện và tính kỵ nước để kháng "Ảo giác sinh học".</span>
              </div>
            </li>
            <li className="flex items-center gap-4">
              <div className="w-12 h-12 bg-rose-500/20 text-rose-400 rounded-xl flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined">bolt</span>
              </div>
              <div>
                <strong className="block text-xl">4. Tối ưu hóa Tài Nguyên</strong>
                <span className="text-sm text-slate-400">Chạy mượt mà trên các thiết bị cá nhân có GPU phổ thông (&lt;24GB).</span>
              </div>
            </li>
          </ul>
        </div>
        <div className="text-center px-8">
          <span className="material-symbols-outlined text-[8rem] text-slate-200 dark:text-slate-800 drop-shadow mb-6">extension</span>
          <p className="text-2xl font-medium leading-relaxed text-slate-700 dark:text-slate-300">
            Mục tiêu là xây dựng khối <strong>Bộ Sinh (Generator)</strong> thông minh, vừa sáng tạo vừa tuân thủ nghiêm ngặt các quy luật hóa học.
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
        <div className="w-full bg-white/60 dark:bg-slate-900/60 rounded-3xl p-6 border border-slate-200 dark:border-slate-700 shadow-sm relative overflow-hidden">
          <div className="flex flex-col md:flex-row justify-between items-center text-center font-bold text-[0.8rem] md:text-sm lg:text-base mb-6 text-slate-500 gap-2">
            <div className="bg-slate-100 dark:bg-slate-800 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-700 grow">Frozen ESM-2</div>
            <span className="material-symbols-outlined text-xl rotate-90 md:rotate-0">add</span>
            <div className="bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 px-4 py-2 rounded-xl border border-purple-200 dark:border-purple-700 grow">GATv2 Graph</div>
            <span className="material-symbols-outlined text-xl rotate-90 md:rotate-0">add</span>
            <div className="bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 px-4 py-2 rounded-xl border border-amber-200 dark:border-amber-700 grow">Cross-Attention</div>
            <span className="material-symbols-outlined text-xl rotate-90 md:rotate-0">arrow_forward</span>
            <div className="bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300 px-4 py-2 rounded-xl border border-rose-200 dark:border-rose-700 grow">Transformer Gen</div>
          </div>
          <div className="relative group w-full mx-auto flex justify-center">
            <img
              src="/Peptide-Design-Lightweight-Model/ESM2-GAT.png"
              alt="Architecture Overview"
              className="relative rounded-2xl shadow object-contain bg-white dark:bg-slate-800 max-h-[480px] w-auto"
            />
          </div>
        </div>
      </div>
    )
  },
  {
    id: 6,
    title: "5. Thành phần 1 — Mô hình nền tảng ESM-2 (Đóng băng)",
    content: (
      <div className="grid md:grid-cols-2 gap-10 items-center h-full">
        <div className="flex justify-center h-full">
          <img src="/Peptide-Design-Lightweight-Model/esm2.png" alt="Kiến trúc ESM-2" className="rounded-3xl border border-slate-200 dark:border-slate-700 shadow-lg object-contain bg-white w-full h-full max-h-[480px]" />
        </div>
        <div className="space-y-6 text-xl">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-8 rounded-3xl border border-blue-200 dark:border-blue-800 shadow-sm">
            <p className="font-semibold text-blue-900 dark:text-blue-100 leading-relaxed mb-6">
              <strong>ESM-2 (650M):</strong> Mã hóa tri thức tiến hóa học được từ 250 triệu chuỗi Protein thuộc cơ sở dữ liệu UniRef50.
            </p>
            <h4 className="font-black text-2xl text-blue-700 dark:text-blue-300 mb-4 border-b border-blue-200 dark:border-blue-800 pb-2 uppercase italic">Chiến lược: Đóng băng tham số</h4>
            <ul className="list-disc pl-6 space-y-4 text-slate-700 dark:text-slate-300 font-medium">
              <li>
                <strong>Đóng băng toàn bộ 33 khối (Blocks):</strong> Giữ nguyên "ngữ pháp" cốt lõi, không cập nhật gradient vào mô hình nền tảng.
              </li>
              <li>
                <strong>Lợi ích:</strong> Tiết kiệm tối đa tài nguyên tính toán, tránh hiện tượng "Quên tri thức cũ" (Catastrophic Forgetting).
              </li>
              <li>Trích xuất các đặc trưng tiềm ẩn (Latent Motifs) để định hình hướng đi cho bộ sinh.</li>
            </ul>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 7,
    title: "6. Thành phần 2 — Bộ mã hóa cấu trúc GATv2",
    content: (
      <div className="grid md:grid-cols-2 gap-10 items-center h-full">
        <div className="space-y-6 text-xl order-2 md:order-1">
          <div className="bg-purple-50 dark:bg-purple-900/20 p-8 rounded-3xl border border-purple-200 dark:border-purple-800 shadow-sm">
            <p className="font-semibold text-purple-900 dark:text-purple-100 leading-relaxed mb-6">
              Giải quyết bài toán <strong>Không gian 3D</strong> thông qua Mạng Đồ thị tương tác (Graph Attention Networks v2).
            </p>
            <ul className="list-disc pl-6 space-y-4 text-slate-700 dark:text-slate-300 font-medium mb-6">
              <li><strong>Đỉnh (Node):</strong> Đại diện cho Amino acid mang đặc trưng hóa học đã nhúng.</li>
              <li><strong>Cạnh (Edge):</strong> Tương tác giữa các lân cận gần nhất (KNN) với bán kính <code>&lt; 8Å</code>.</li>
              <li><strong>Cơ chế Chú ý động:</strong> GATv2 giúp học các tương tác phi tuyến phức tạp giữa các cụm amino acid xa nhau về vị trí chuỗi nhưng gần nhau trong không gian.</li>
            </ul>
          </div>
        </div>
        <div className="flex justify-center h-full order-1 md:order-2">
          <img src="/Peptide-Design-Lightweight-Model/gatarchi.png" alt="Kiến trúc GATv2" className="rounded-3xl border border-slate-200 dark:border-slate-700 shadow-lg object-contain bg-white p-4 w-full h-full max-h-[480px]" />
        </div>
      </div>
    )
  },
  {
    id: 8,
    title: "7. Hợp nhất Đa phương thức & Điều kiện Hóa-Lý",
    content: (
      <div className="space-y-6 h-full flex flex-col justify-center">
        <div className="text-center max-w-4xl mx-auto mb-6">
          <h3 className="text-2xl font-black text-amber-600 dark:text-amber-400 mb-2 uppercase">Hợp nhất bằng cơ chế Cross-Attention</h3>
          <p className="text-lg font-medium text-slate-600 dark:text-slate-300">
            Hợp nhất <strong>Bản đồ Cấu trúc (3D)</strong> và <strong>Ngữ pháp Chuỗi (1D)</strong> cùng 18 tham số điều kiện đặc trưng.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          {[
            'Chỉ số Ổn định', 'Hoạt tính Sinh học', 'Độ Độc tính', 'Chỉ số Aliphatic',
            'Khối lượng Phân tử', 'Độ Kỵ nước (GRAVY)', 'Điện tích Thuần', 'Tính Thơm'
          ].map((attr, idx) => (
            <div key={idx} className="bg-amber-100/50 dark:bg-amber-900/20 p-3 rounded-xl border border-amber-200 dark:border-amber-800 text-sm font-bold text-center text-amber-800 dark:text-amber-200">
              {attr}
            </div>
          ))}
        </div>

        <div className="bg-slate-800 text-slate-200 p-8 rounded-3xl shadow-xl space-y-4">
          <p className="text-lg font-medium italic text-center">
            "Việc nhúng trực tiếp các hằng số Hóa-Lý vào Generator thông qua **Cross-Attention** giúp mô hình luôn bị ràng buộc bởi các tiêu chuẩn ổn định sinh học."
          </p>
          <div className="flex items-center gap-4 text-emerald-400 font-mono text-xs md:text-sm overflow-x-auto whitespace-nowrap pb-2 justify-center">
            <span>Vector Tiềm ẩn (128d)</span>
            <span className="material-symbols-outlined">arrow_forward</span>
            <span>Bản đồ Điều kiện (18d)</span>
            <span className="material-symbols-outlined">arrow_forward</span>
            <span>Hợp nhất Đa phương thức</span>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 9,
    title: "8. Thành phần 3 — Lõi tạo sinh (Transformer-GAN)",
    content: (
      <div className="grid md:grid-cols-2 gap-8 h-full">
        <div className="bg-rose-50 dark:bg-rose-900/10 p-8 rounded-3xl border border-rose-200 dark:border-rose-900 shadow-sm relative overflow-hidden flex flex-col justify-center">
          <h3 className="text-3xl font-black text-rose-600 dark:text-rose-400 mb-6 uppercase">Bộ Sinh (Generator)</h3>
          <ul className="space-y-5 text-xl font-medium text-slate-700 dark:text-slate-300">
            <li><strong>Transformer Decoder:</strong> Sinh chuỗi peptide linh hoạt dựa trên kiến thức từ bước hợp nhất.</li>
            <li><strong>Gumbel-Softmax:</strong> Kỹ thuật cho phép lan truyền ngược qua các token rời rạc trong huấn luyện đối kháng.</li>
            <li><strong>Nucleus Sampling:</strong> Tối ưu hóa tính đa dạng sinh học của tập dữ liệu tạo ra.</li>
          </ul>
        </div>

        <div className="bg-slate-100 dark:bg-slate-800 p-8 rounded-3xl border border-slate-300 dark:border-slate-700 shadow-sm relative overflow-hidden flex flex-col justify-center">
          <h3 className="text-3xl font-black text-slate-800 dark:text-slate-100 mb-6 uppercase">Bộ Phân Biệt (WGAN-GP)</h3>
          <ul className="space-y-5 text-xl font-medium text-slate-600 dark:text-slate-400">
            <li><strong>CNN 1D Discriminator:</strong> Nhận diện các đặc trưng sinh học trên bề mặt chuỗi.</li>
            <li><strong>Wasserstein Loss:</strong> Đo lường khoảng cách phân phối mượt mà, tránh hiện tượng lỗi hội tụ (Mode collapse).</li>
            <li><strong>Gradient Penalty:</strong> Đảm bảo tính ổn định và hội tụ nhanh cho mạng GAN.</li>
          </ul>
        </div>
      </div>
    )
  },
  {
    id: 10,
    title: "9. Tối ưu hóa bằng Học tăng cường (RL)",
    content: (
      <div className="flex flex-col h-full bg-emerald-50 dark:bg-emerald-900/10 rounded-3xl border border-emerald-200 dark:border-emerald-800 p-8 shadow-sm">
        <h3 className="text-2xl font-black text-emerald-700 dark:text-emerald-400 mb-6 border-b border-emerald-200 dark:border-emerald-800 pb-3 flex items-center gap-2 uppercase">
          <span className="material-symbols-outlined text-3xl">sports_esports</span> Chiến lược Tự phê bình (SCST)
        </h3>
        <p className="text-xl font-medium text-slate-700 dark:text-slate-300 mb-8 leading-relaxed">
          Sử dụng <strong>Self-Critical Sequence Training (SCST)</strong> để tinh chỉnh mô hình dựa trên các chỉ số sinh học thực tế.
        </p>

        <div className="grid md:grid-cols-2 gap-8 text-lg flex-1">
          <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-emerald-100 dark:border-emerald-900 flex flex-col justify-center shadow">
            <h4 className="font-bold text-emerald-600 mb-4 text-xl uppercase">Hàm Thưởng (Reward Function)</h4>
            <ul className="space-y-3 font-semibold text-slate-700 dark:text-slate-300">
              <li className="flex items-center gap-2">✅ <strong>R_ổn định:</strong> Thưởng cao cho các chuỗi có Instability Index &lt; 40.</li>
              <li className="flex items-center gap-2">✅ <strong>R_kháng khuẩn:</strong> Thưởng cho peptide có xác suất kháng khuẩn dự đoán cao.</li>
              <li className="flex items-center gap-2">❌ <strong>R_độc tính:</strong> Phạt nặng các chuỗi peptide có độ độc tính (Hemolytic) cao.</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-emerald-100 dark:border-emerald-900 flex flex-col justify-center shadow">
            <h4 className="font-bold text-emerald-600 mb-4 text-xl uppercase">Mục tiêu tối ưu</h4>
            <p className="text-slate-600 dark:text-slate-400 italic">
              Thúc đẩy mô hình vượt qua giới hạn của dữ liệu huấn luyện, đạt tới vùng không gian "tối ưu sinh học" mà thực nghiệm yêu cầu.
            </p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 11,
    title: "10. Quy trình huấn luyện 3 giai đoạn",
    content: (
      <div className="space-y-6 text-xl h-full flex flex-col justify-center">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative max-w-6xl mx-auto w-full">
          <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-md border border-slate-200 dark:border-slate-700 flex flex-col items-center text-center gap-4 transition hover:-translate-y-1">
            <div className="bg-blue-100 dark:bg-blue-900/80 text-blue-600 dark:text-blue-400 font-black text-3xl h-14 w-14 rounded-full flex items-center justify-center shadow-sm">1</div>
            <h3 className="font-black text-xl text-blue-600 uppercase">Tiền huấn luyện (MLE)</h3>
            <p className="text-base text-slate-600 dark:text-slate-300 font-medium">Học phân phối dữ liệu gốc qua phương pháp Teacher Forcing. Thiết lập nền tảng ngữ pháp peptide.</p>
          </div>

          <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-md border border-slate-200 dark:border-slate-700 flex flex-col items-center text-center gap-4 transition hover:-translate-y-1">
            <div className="bg-purple-100 dark:bg-purple-900/80 text-purple-600 dark:text-purple-400 font-black text-3xl h-14 w-14 rounded-full flex items-center justify-center shadow-sm">2</div>
            <h3 className="font-black text-xl text-purple-600 uppercase">Đối kháng (GAN)</h3>
            <p className="text-base text-slate-600 dark:text-slate-300 font-medium">Huấn luyện đối kháng WGAN-GP. Nâng cao tính thực tế và đa dạng của các chuỗi được tạo ra.</p>
          </div>

          <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-md border border-slate-200 dark:border-slate-700 flex flex-col items-center text-center gap-4 transition hover:-translate-y-1">
            <div className="bg-emerald-100 dark:bg-emerald-900/80 text-emerald-600 dark:text-emerald-400 font-black text-3xl h-14 w-14 rounded-full flex items-center justify-center shadow-sm">3</div>
            <h3 className="font-black text-xl text-emerald-600 uppercase">Học tăng cường (RL)</h3>
            <p className="text-base text-slate-600 dark:text-slate-300 font-medium">Tinh chỉnh bằng SCST. Ép mô hình hội tụ về các vùng peptide có hoạt tính sinh học cao nhất.</p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 12,
    title: "11. Kết quả thực nghiệm và Thảo luận",
    content: (
      <div className="space-y-6 flex flex-col justify-center h-full">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {[
            { label: 'Tỉ lệ Ổn định', value: '71.2%', desc: 'II < 40' },
            { label: 'Khả năng Kháng khuẩn', value: '73.6%', desc: 'Xác suất > 0.5' },
            { label: 'Tính Mới', value: '100%', desc: 'Chuỗi chưa từng xuất hiện' },
            { label: 'Tính Đa dạng', value: '0.94', desc: 'Bigram Entropy' }
          ].map((stat, i) => (
            <div key={i} className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm text-center">
              <span className="block text-sm font-bold text-slate-500 uppercase mb-1">{stat.label}</span>
              <span className="block text-3xl font-black text-blue-600 dark:text-blue-400">{stat.value}</span>
              <span className="block text-xs text-slate-400 mt-1">{stat.desc}</span>
            </div>
          ))}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-4 bg-white/50 dark:bg-slate-800/50 rounded-2xl border border-slate-200 dark:border-slate-700 text-center">
            <h4 className="font-bold text-emerald-600 dark:text-emerald-400 mb-2 uppercase">Độ ổn định</h4>
            <img src="/Peptide-Design-Lightweight-Model/instability_hist.png" alt="Biểu đồ Chỉ số Ổn định" className="w-full h-auto rounded-lg mb-2 border border-slate-200 dark:border-slate-600 aspect-video object-cover" />
          </div>
          <div className="p-4 bg-white/50 dark:bg-slate-800/50 rounded-2xl border border-slate-200 dark:border-slate-700 text-center">
            <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-2 uppercase">Độc tính</h4>
            <img src="/Peptide-Design-Lightweight-Model/hemolytic_score_hist.png" alt="Biểu đồ Độc tính (Hemolytic)" className="w-full h-auto rounded-lg mb-2 border border-slate-200 dark:border-slate-600 aspect-video object-cover" />
          </div>
          <div className="p-4 bg-white/50 dark:bg-slate-800/50 rounded-2xl border border-slate-200 dark:border-slate-700 text-center">
            <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-2 uppercase">Độ Kỵ nước</h4>
            <img src="/Peptide-Design-Lightweight-Model/gravy_hist.png" alt="Biểu đồ Độ Kỵ nước (GRAVY)" className="w-full h-auto rounded-lg mb-2 border border-slate-200 dark:border-slate-600 aspect-video object-cover" />
          </div>
        </div>
      </div>
    )
  },
  {
    id: 13,
    title: "12. Đóng góp và Ý nghĩa khoa học",
    content: (
      <div className="bg-indigo-50 dark:bg-indigo-900/10 p-10 rounded-3xl border border-indigo-200 dark:border-indigo-900/50 shadow-sm h-full flex flex-col justify-center">
        <div className="grid md:grid-cols-2 gap-8 text-xl font-medium text-slate-800 dark:text-slate-200">
          <div className="flex gap-4 p-5 bg-white/60 dark:bg-slate-800/60 rounded-2xl shadow-sm border border-indigo-100 dark:border-indigo-900/50">
            <span className="material-symbols-outlined text-4xl text-indigo-500">hub</span>
            <div><strong className="block text-indigo-700 dark:text-indigo-300 mb-1">Kiến trúc Lai Đột phá</strong>
              Tận dụng tri thức từ các mô hình ngôn ngữ lớn (PLMs - ESM-2) nhưng vẫn duy trì sự gọn nhẹ nhờ đóng băng tham số.</div>
          </div>
          <div className="flex gap-4 p-5 bg-white/60 dark:bg-slate-800/60 rounded-2xl shadow-sm border border-indigo-100 dark:border-indigo-900/50">
            <span className="material-symbols-outlined text-4xl text-emerald-500">analytics</span>
            <div><strong className="block text-emerald-700 dark:text-emerald-300 mb-1">Kiểm soát Sinh học Chặt chẽ</strong>
              Nhúng trực tiếp các ràng buộc Hóa-Lý vào không gian tiềm ẩn thông qua kỹ thuật Cross-Attention.</div>
          </div>
          <div className="flex gap-4 p-5 bg-white/60 dark:bg-slate-800/60 rounded-2xl shadow-sm border border-indigo-100 dark:border-indigo-900/50">
            <span className="material-symbols-outlined text-4xl text-blue-500">bolt</span>
            <div><strong className="block text-blue-700 dark:text-blue-300 mb-1">Hiệu năng Vượt trội</strong>
              Khả năng sinh tạo hàng triệu chuỗi peptide với độ ổn định cao trên phần cứng phổ thông.</div>
          </div>
          <div className="flex gap-4 p-5 bg-white/60 dark:bg-slate-800/60 rounded-2xl shadow-sm border border-indigo-100 dark:border-indigo-900/50">
            <span className="material-symbols-outlined text-4xl text-rose-500">science</span>
            <div><strong className="block text-rose-700 dark:text-rose-300 mb-1">Định hướng Y học Thực tiễn</strong>
              Cung cấp các ứng viên peptide chất lượng cao, rút ngắn thời gian nghiên cứu kháng sinh thế hệ mới.</div>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 14,
    title: "13. Thách thức và Hạn chế",
    content: (
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 h-full items-center">
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[280px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">view_in_ar</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Mô phỏng Đồ thị Tĩnh</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">GATv2 chỉ mô tả liên kết tại một trạng thái tĩnh, chưa tính đến các biến động động lực học phân tử theo thời gian.</p>
        </div>
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[280px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">function</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Đánh giá In-silico</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Các hàm thưởng RL dựa trên tính toán mô phỏng, cần được kiểm chứng thêm thông qua thực nghiệm phòng lab (wet-lab).</p>
        </div>
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[280px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">layers</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Giới hạn Độ dài</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Mô hình hiện tại tối ưu nhất cho peptide ngắn (&lt; 50 amino acid), hiệu năng có thể giảm với các chuỗi protein lớn.</p>
        </div>
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[280px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">database</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Sự đa dạng của Dữ liệu</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Kết quả phụ thuộc lớn vào sự phong phú và độ chính xác của các tập dữ liệu peptide kháng khuẩn công khai.</p>
        </div>
      </div>
    )
  },
  {
    id: 15,
    title: "14. Tầm nhìn và Hướng phát triển",
    content: (
      <div className="flex flex-col justify-center h-full">
        <div className="max-w-4xl mx-auto w-full space-y-4">
          <div className="bg-white/60 dark:bg-slate-800/60 p-5 rounded-2xl border border-slate-200 dark:border-slate-700 flex gap-4 items-center">
            <div className="bg-emerald-500 p-2 rounded-xl text-white flex shrink-0"><span className="material-symbols-outlined font-bold">account_tree</span></div>
            <p className="text-lg font-bold text-slate-700 dark:text-slate-200">Tích hợp các mô hình dự đoán cấu trúc 3D trực tiếp (AlphaFold3) vào quá trình tối ưu hóa.</p>
          </div>
          <div className="bg-white/60 dark:bg-slate-800/60 p-5 rounded-2xl border border-slate-200 dark:border-slate-700 flex gap-4 items-center">
            <div className="bg-emerald-500 p-2 rounded-xl text-white flex shrink-0"><span className="material-symbols-outlined font-bold">rocket_launch</span></div>
            <p className="text-lg font-bold text-slate-700 dark:text-slate-200">Phát triển các dòng peptide đặc trị cho từng chủng vi khuẩn kháng thuốc cụ thể.</p>
          </div>
          <div className="bg-white/60 dark:bg-slate-800/60 p-5 rounded-2xl border border-slate-200 dark:border-slate-700 flex gap-4 items-center">
            <div className="bg-emerald-500 p-2 rounded-xl text-white flex shrink-0"><span className="material-symbols-outlined font-bold">water_drop</span></div>
            <p className="text-lg font-bold text-slate-700 dark:text-slate-200">Ứng dụng mô hình khuếch tán (Diffusion Models) để tinh chỉnh chi tiết cấu trúc peptide.</p>
          </div>
          <div className="bg-white/60 dark:bg-slate-800/60 p-5 rounded-2xl border border-slate-200 dark:border-slate-700 flex gap-4 items-center shadow-md">
            <div className="bg-blue-600 p-2 rounded-xl text-white flex shrink-0"><span className="material-symbols-outlined font-bold">language</span></div>
            <p className="text-lg font-bold text-blue-800 dark:text-blue-300">Triển khai hệ thống SaaS hỗ trợ cộng đồng nghiên cứu peptide kháng khuẩn toàn cầu.</p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 16,
    title: "15. Hướng dẫn sử dụng & Demo",
    content: (
      <div className="flex flex-col items-center justify-center h-full text-center space-y-10">
        <h2 className="text-5xl md:text-[4rem] font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 via-blue-500 to-purple-600 pb-2 mb-2 uppercase">Kết thúc báo cáo</h2>
        <p className="text-2xl font-medium text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Cảm ơn Hội đồng đã dành thời gian theo dõi. Mời quý Thầy Cô và các bạn cùng trải nghiệm hệ thống tại đây.
        </p>
        <a
          href="/Peptide-Design-Lightweight-Model/generation"
          className="group relative px-12 py-5 font-bold text-white rounded-full bg-slate-900 dark:bg-white dark:text-slate-900 text-2xl overflow-hidden hover:scale-105 transition-transform duration-300 shadow-2xl flex items-center justify-center gap-3"
        >
          <div className="absolute inset-0 w-full h-full border-[6px] border-emerald-500/30 rounded-full blur-[10px] scale-110 group-hover:blur-[20px] transition-all"></div>
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-emerald-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <span className="relative z-10 flex items-center gap-3">
            Bắt đầu Trải nghiệm
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
