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
          Nghiên cứu và xây dựng mô hình Lightweight cho sinh tạo chuỗi Peptide ngắn có độ ổn định cấu trúc cao
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
    title: "GIỚI THIỆU",
    content: (
      <div className="grid md:grid-cols-2 gap-10 text-xl items-center h-full">
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-8 rounded-3xl border border-blue-200 dark:border-blue-800 shadow-sm relative">
            <h3 className="font-bold text-2xl text-blue-600 dark:text-blue-400 mb-4 flex items-center gap-2 border-b-2 border-blue-200 dark:border-blue-800 pb-3">
              <span className="material-symbols-outlined text-3xl">biotech</span> Peptide
            </h3>
            <p className="font-medium text-slate-700 dark:text-slate-300 leading-relaxed">
              Peptide là chuỗi amino acid ngắn từ 2-50.<br /> Peptide kháng khuẩn (AMPs), hay peptide bảo vệ vật chủ (HDPs), là các chuỗi acid amin ngắn thuộc hệ miễn dịch bẩm sinh của sinh vật,có khả năng tiêu diệt rộng vi khuẩn, virus, nấm và tế bào ung thư bằng cách phá hủy màng tế bào. Được xem là giải pháp thay thế kháng sinh tiềm năng và đối phó với tình trạng Kháng kháng sinh (AMR).
            </p>
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
              Vượt xa số lượng nguyên tử trong vũ trụ. Việc tìm kiếm trong không gian này bắt buộc cần đến sức mạnh của Trí tuệ nhân tạo Tạo sinh.
            </p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 3,
    title: "VẤN ĐỀ NGHIÊN CỨU",
    content: (
      <div className="grid md:grid-cols-2 gap-8 h-full items-center">
        <div className="space-y-6">
          <div className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm">
            <h4 className="font-black text-2xl text-slate-800 dark:text-slate-100 mb-4">Vấn đề của các mô hình hiện nay</h4>
            <ul className="space-y-4 text-lg">
              <li className="flex gap-3">
                <span className="text-rose-500 material-symbols-outlined shrink-0">error</span>
                <span><strong>Ảo giác cấu trúc:</strong> Sinh chuỗi đúng ngữ pháp nhưng không thể tồn tại ổn định trong 3D.</span>
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
            "Thiết kế hệ thống sinh tạo peptide tuân thủ các quy luật hóa-lý thông qua tích hợp đặc trưng đa phương thức và tối ưu hóa dựa trên phần thưởng. Đảm bảo các ứng viên có cấu trúc ổn định, điện tích phù hợp và tính thực tiễn cao trong ứng dụng y sinh."
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/10 p-4 rounded-xl">
              <span className="block text-2xl font-bold">II &lt; 40</span>
              <span className="text-xs">Độ ổn định cấu trúc</span>
            </div>
            <div className="bg-white/10 p-4 rounded-xl">
              <span className="block text-2xl font-bold">P &ge; 0.7</span>
              <span className="text-xs">Hoạt tính kháng khuẩn</span>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 4,
    title: "YÊU CẦU CẦN THIẾT CỦA KIẾN TRÚC",
    content: (
      <div className="flex flex-col gap-10 h-full items-center justify-center">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 w-full">
          <div className="bg-slate-800 text-slate-100 p-6 rounded-2xl shadow-lg border border-slate-700 hover:scale-105 transition-transform flex flex-col items-center text-center">
            <span className="material-symbols-outlined text-4xl text-blue-400 mb-4">auto_stories</span>
            <strong className="block text-lg text-blue-400 mb-2">Kế thừa tri thức</strong>
            <span className="text-xs text-slate-300 leading-relaxed">Dùng ESM-2 (Freezing) để trích xuất motif sinh học, giữ ngữ pháp protein ổn định.</span>
          </div>
          <div className="bg-slate-800 text-slate-100 p-6 rounded-2xl shadow-lg border border-slate-700 hover:scale-105 transition-transform flex flex-col items-center text-center">
            <span className="material-symbols-outlined text-4xl text-emerald-400 mb-4">shape_line</span>
            <strong className="block text-lg text-emerald-400 mb-2">Am hiểu Cấu trúc</strong>
            <span className="text-xs text-slate-300 leading-relaxed">Mạng GATv2 mô phỏng tương tác 3D Cα, đảm bảo tính bền vững vật lý cho chuỗi.</span>
          </div>
          <div className="bg-slate-800 text-slate-100 p-6 rounded-2xl shadow-lg border border-slate-700 hover:scale-105 transition-transform flex flex-col items-center text-center">
            <span className="material-symbols-outlined text-4xl text-amber-400 mb-4">science</span>
            <strong className="block text-lg text-amber-400 mb-2">Kiểm soát Đặc tính</strong>
            <span className="text-xs text-slate-300 leading-relaxed">Nhúng II, GRAVY, Charge qua Cross-Attention để kiểm soát sinh học chặt chẽ.</span>
          </div>
          <div className="bg-slate-800 text-slate-100 p-6 rounded-2xl shadow-lg border border-slate-700 hover:scale-105 transition-transform flex flex-col items-center text-center">
            <span className="material-symbols-outlined text-4xl text-rose-400 mb-4">bolt</span>
            <strong className="block text-lg text-rose-400 mb-2">Tối ưu Tài Nguyên</strong>
            <span className="text-xs text-slate-300 leading-relaxed">Thiết kế Lightweight chạy mượt trên GPU cá nhân, không cần hạ tầng đắt đỏ.</span>
          </div>
        </div>

        <div className="flex flex-col items-center gap-4 text-slate-400">
          <span className="material-symbols-outlined text-4xl animate-bounce">keyboard_double_arrow_down</span>
          <div className="bg-blue-600/10 dark:bg-blue-900/10 p-8 rounded-[40px] border-4 border-dashed border-blue-500/30 max-w-4xl text-center shadow-inner relative group">
            <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-blue-600 px-6 py-2 rounded-full text-white font-black text-sm uppercase tracking-widest shadow-lg">Yêu cầu Tổng quát</div>
            <p className="text-2xl font-black leading-relaxed text-blue-800 dark:text-blue-300">
              Đảm bảo peptide sinh ra tuân thủ các quy luật hóa học bằng cách tích hợp đặc trưng hóa-lý và áp dụng ràng buộc đa mục tiêu. Cơ chế tối ưu hóa phần thưởng giúp định hình các chuỗi có cấu trúc ổn định, điện tích tối ưu và tính thực tiễn cao.
            </p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 5,
    title: "TỔNG QUAN KIẾN TRÚC MÔ HÌNH",
    content: (
      <div className="flex flex-col items-center justify-center gap-10 h-full w-full max-w-6xl mx-auto py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full items-center relative">
          {/* Input Layer */}
          <div className="flex flex-col gap-6">
            <div className="bg-blue-600/10 dark:bg-blue-900/20 p-8 rounded-[40px] border-4 border-blue-600/30 flex flex-col items-center gap-4 shadow-lg text-center relative group hover:scale-105 transition-all">
              <div className="absolute -top-4 bg-blue-600 text-white px-5 py-1 rounded-full text-[0.6rem] font-black uppercase tracking-widest shadow-md">Input Encoder 1D</div>
              <span className="material-symbols-outlined text-5xl text-blue-600">dna</span>
              <div className="space-y-1">
                <h4 className="font-black text-xl text-blue-800 dark:text-blue-200">ESM-2 Backbone</h4>
                <p className="text-xs font-semibold text-slate-500">Mã hóa ngữ pháp chuỗi 1D (Frozen 650M)</p>
              </div>
            </div>

            <div className="bg-purple-600/10 dark:bg-purple-900/20 p-8 rounded-[40px] border-4 border-purple-600/30 flex flex-col items-center gap-4 shadow-lg text-center relative group hover:scale-105 transition-all">
              <div className="absolute -top-4 bg-purple-600 text-white px-5 py-1 rounded-full text-[0.6rem] font-black uppercase tracking-widest shadow-md">Input Encoder 3D</div>
              <span className="material-symbols-outlined text-5xl text-purple-600">hub</span>
              <div className="space-y-1">
                <h4 className="font-black text-xl text-purple-800 dark:text-purple-200">GATv2 Graph</h4>
                <p className="text-xs font-semibold text-slate-500">Mô phỏng tương tác 3D nguyên tử C&alpha;</p>
              </div>
            </div>
          </div>

          {/* Fusion Layer */}
          <div className="flex flex-col items-center justify-center relative">
            {/* Connectors from Left */}
            <div className="hidden md:block absolute -left-12 top-1/4 w-12 h-0.5 bg-gradient-to-r from-blue-600 to-amber-600 opacity-30"></div>
            <div className="hidden md:block absolute -left-12 bottom-1/4 w-12 h-0.5 bg-gradient-to-r from-purple-600 to-amber-600 opacity-30"></div>

            <div className="bg-amber-500/10 dark:bg-amber-900/20 p-10 rounded-full border-4 border-amber-500/40 flex flex-col items-center gap-4 shadow-2xl text-center relative z-10 w-full max-w-[280px] aspect-square justify-center group hover:scale-110 transition-all border-dashed animate-spin-slow">
              <div className="absolute -top-4 bg-amber-500 text-white px-5 py-1 rounded-full text-[0.7rem] font-black uppercase tracking-widest shadow-md animate-none">Multimodal Bridge</div>
              <span className="material-symbols-outlined text-6xl text-amber-500 animate-pulse">sync_alt</span>
              <div className="space-y-1">
                <h4 className="font-black text-2xl text-amber-800 dark:text-amber-200">Cross-Attention</h4>
                <p className="text-[0.7rem] font-bold text-slate-500 px-4">Hợp nhất ngữ cảnh 1D/3D & 18 chỉ số Điều kiện</p>
              </div>
            </div>

            {/* Connector to Right */}
            <div className="hidden md:block absolute -right-12 top-1/2 w-12 h-1 bg-gradient-to-r from-amber-500 to-rose-500 opacity-50"></div>
          </div>

          {/* Generator Layer */}
          <div className="flex flex-col gap-6">
            <div className="bg-rose-600/10 dark:bg-rose-900/20 p-10 rounded-[50px] border-4 border-rose-600/40 flex flex-col items-center gap-6 shadow-2xl text-center relative group hover:scale-105 transition-all">
              <div className="absolute -top-4 bg-rose-600 text-white px-6 py-2 rounded-full text-xs font-black uppercase tracking-widest shadow-lg">Final Generator</div>
              <span className="material-symbols-outlined text-7xl text-rose-600">auto_awesome</span>
              <div className="space-y-2">
                <h4 className="font-black text-3xl text-rose-800 dark:text-rose-200 tracking-tight">Transformer-GAN</h4>
                <p className="text-sm font-bold text-slate-600 dark:text-slate-400">Sinh chuỗi Peptide (Gumbel-Softmax)</p>
              </div>
              <div className="mt-4 pt-4 border-t border-rose-200 dark:border-rose-800 flex items-center gap-3 text-emerald-600 font-black">
                <span className="material-symbols-outlined">check_circle</span>
                <span className="uppercase text-xs tracking-widest">Output: Optimized Peptide</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-slate-900/5 dark:bg-white/5 p-6 rounded-3xl border border-slate-200 dark:border-slate-800 max-w-5xl w-full text-center">
          <p className="text-lg font-medium text-slate-500 italic leading-relaxed">
            Mô hình kết hợp Tri thức tiến hóa (1D) và Bản đồ lân cận nguyên tử (3D) thông qua cơ chế Chú ý chéo (Cross-Attention), tạo ra một bộ sinh (Generator) có khả năng dự đoán cấu trúc peptide chuẩn xác với độ ổn định sinh học cao.
          </p>
        </div>
      </div>
    )
  },
  {
    id: 6,
    title: "ESM-2 BACKBONE",
    content: (
      <div className="grid md:grid-cols-2 gap-10 items-center h-full relative">
        <div className="absolute top-0 right-0 bg-blue-600 text-white px-4 py-1 rounded-full text-xs font-black uppercase shadow-md z-20">Pre-trained Knowledge</div>
        <div className="flex justify-center h-full">
          <img src="/Peptide-Design-Lightweight-Model/esm2.png" alt="Kiến trúc ESM-2" className="rounded-3xl border border-slate-200 dark:border-slate-700 shadow-lg object-contain bg-white w-full h-full max-h-[480px]" />
        </div>
        <div className="space-y-6 text-xl">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-8 rounded-3xl border border-blue-200 dark:border-blue-800 shadow-sm">
            <p className="font-semibold text-blue-900 dark:text-blue-100 leading-relaxed mb-6">
              <strong>ESM-2 (Evolutionary Scale Modeling):</strong> Mô hình ngôn ngữ protein dựa trên kiến trúc Transformer (BERT-style), được huấn luyện trên hàng triệu trình tự axit amin để dự đoán cấu trúc và chức năng.
            </p>
            <h4 className="font-black text-2xl text-blue-700 dark:text-blue-300 mb-4 border-b border-blue-200 dark:border-blue-800 pb-2 uppercase italic">Đóng băng tham số </h4>
            <ul className="list-disc pl-6 space-y-4 text-slate-700 dark:text-slate-300 font-medium">
              <li>
                Đóng băng toàn bộ 33 lớp để kế thừa khả năng hiểu ngữ pháp sinh học khổng lồ mà không tốn tài nguyên huấn luyện lại.
              </li>
              <li>
                Lấy các đặc trưng tiềm ẩn từ lớp cuối cùng để làm nền tảng cho bộ sinh Generator.
              </li>
              <li>
                Tiết kiệm GPU giúp mô hình chạy mượt mà trên phần cứng cá nhân.
              </li>
            </ul>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 7,
    title: "BỘ MÃ HÓA CẤU TRÚC GATV2",
    content: (
      <div className="grid md:grid-cols-2 gap-10 items-center h-full">
        <div className="space-y-6 text-xl order-2 md:order-1">
          <div className="bg-purple-50 dark:bg-purple-900/20 p-8 rounded-3xl border border-purple-200 dark:border-purple-800 shadow-sm">
            <p className="font-semibold text-purple-900 dark:text-purple-100 leading-relaxed mb-6">
              Giải quyết bài toán <strong>Không gian 3D</strong> thông qua Mạng Đồ thị tương tác.
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
    title: "HỢP NHẤT ĐA PHƯƠNG THỨC VÀ ĐIỀU KIỆN HÓA LÝ",
    content: (
      <div className="space-y-6 h-full flex flex-col justify-center">
        <div className="text-center max-w-4xl mx-auto mb-6">
          <h3 className="text-2xl font-black text-amber-600 dark:text-amber-400 mb-2 uppercase">Hợp nhất bằng cơ chế Cross-Attention</h3>
        </div>

        <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-6">
          {[
            'Stability', 'AMP Prob', 'Toxicity', 'Aliphatic', 'Mol. Weight', 'GRAVY',
            'Net Charge', 'Isoelectric', 'Aromaticity', 'Flexibility', 'Solubility', 'Boman Index',
            'Hydro. Moment', 'Helix Prop.', 'Beta Prop.', 'Turn Prop.', 'Extinction', 'Cys %'
          ].map((attr, idx) => (
            <div key={idx} className="bg-amber-100/50 dark:bg-amber-900/20 p-2 rounded-xl border border-amber-200 dark:border-amber-800 text-[0.7rem] font-black text-center text-amber-800 dark:text-amber-200 uppercase tracking-tighter shadow-sm">
              {attr}
            </div>
          ))}
        </div>

        <div className="bg-slate-800 text-slate-200 p-8 rounded-3xl shadow-xl space-y-4">
          <p className="text-xl font-medium italic text-center leading-relaxed">
            Hợp nhất <strong>Cấu trúc Không gian (GATv2)</strong> và <strong>Ngữ pháp Chuỗi (ESM-2)</strong> cùng 18 tham số điều kiện đặc trưng. Việc nhúng trực tiếp bằng Cross-Attention giúp mô hình luôn bị ràng buộc bởi các quy luật hóa-lý và tiêu chuẩn ổn định sinh học.
          </p>
          <div className="flex items-center gap-4 text-emerald-400 font-mono text-xs md:text-sm overflow-x-auto whitespace-nowrap pb-2 justify-center">
            <span>3D Graph Latent</span>
            <span className="material-symbols-outlined text-sm">add_circle</span>
            <span>1D Sequence Latent</span>
            <span className="material-symbols-outlined text-sm">add_circle</span>
            <span>18 Physicochemical Projections</span>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 9,
    title: "MLE & GAN",
    content: (
      <div className="grid md:grid-cols-2 gap-8 h-full relative">
        <div className="bg-rose-50 dark:bg-rose-900/10 p-8 rounded-3xl border border-rose-200 dark:border-rose-900 shadow-sm relative overflow-hidden flex flex-col justify-center">
          <h3 className="text-3xl font-black text-rose-600 dark:text-rose-400 mb-6 uppercase">LÕI SINH TẠO </h3>
          <ul className="space-y-4 text-xl font-medium text-slate-700 dark:text-slate-300">
            <li><strong>MLE (Pre-training):</strong> Học phân phối peptide qua phương pháp Teacher Forcing để nắm vững ngữ pháp sinh học.</li>
            <li><strong>WGAN-GP:</strong> Huấn luyện đối kháng giúp mô hình sinh ra các chuỗi có tính thực tế và đa dạng cao.</li>
            <li><strong>Gumbel-Softmax:</strong> Lan truyền ngược qua các token rời rạc.</li>
          </ul>
        </div>

        <div className="bg-slate-100 dark:bg-slate-800 p-8 rounded-3xl border border-slate-300 dark:border-slate-700 shadow-sm relative overflow-hidden flex flex-col justify-center">
          <h3 className="text-3xl font-black text-slate-800 dark:text-slate-100 mb-6 uppercase">LÕI PHÂN BIỆT </h3>
          <ul className="space-y-5 text-xl font-medium text-slate-600 dark:text-slate-400">
            <li><strong>CNN 1D:</strong> Trích xuất các đặc trưng bề mặt chuỗi để phân biệt peptide thực và giả.</li>
            <li><strong>Wasserstein Loss:</strong> Đo lường khoảng cách phân phối, tránh hiện tượng mô hình hội tụ quá sớm.</li>
          </ul>
        </div>
      </div>
    )
  },
  {
    id: 10,
    title: "HỌC TĂNG CƯỜNG",
    content: (
      <div className="flex flex-col h-full gap-8">
        <p className="text-lg font-medium text-slate-700 dark:text-slate-300 bg-emerald-100/50 dark:bg-emerald-900/20 p-4 rounded-2xl border border-emerald-200 dark:border-emerald-800/50 mb-2">
          <strong>SCST (Self-Critical Sequence Training)</strong> là một phương pháp trong reinforcement learning dùng để tối ưu các mô hình sinh chuỗi (sequence generation), đặc biệt phổ biến trong NLP và generative models.
        </p>
        <div className="grid md:grid-cols-2 gap-8 flex-1">
          {/* Vòng lặp phản hồi (Feedback Loop) */}
          <div className="bg-slate-900 text-slate-100 p-8 rounded-3xl shadow-xl flex flex-col justify-center relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full -mr-10 -mt-10 blur-2xl group-hover:bg-emerald-500/20 transition-all"></div>
            <h4 className="font-black text-emerald-400 mb-8 border-l-4 border-emerald-500 pl-4 uppercase tracking-widest text-lg">Feedback Loop (SCST)</h4>

            <div className="space-y-6 relative z-10">
              <div className="flex items-center gap-4 bg-white/5 p-4 rounded-2xl border border-white/10">
                <div className="bg-rose-500/20 p-2 rounded-lg text-rose-400"><span className="material-symbols-outlined">auto_awesome</span></div>
                <div>
                  <span className="block text-xs uppercase font-black text-slate-500">Generator</span>
                  <span className="text-sm font-bold">Sinh chuỗi Peptide ngẫu nhiên (Sampling)</span>
                </div>
              </div>

              <div className="flex justify-center"><span className="material-symbols-outlined text-slate-600 animate-bounce">arrow_downward</span></div>

              <div className="flex items-center gap-4 bg-white/5 p-4 rounded-2xl border border-white/10">
                <div className="bg-amber-500/20 p-2 rounded-lg text-amber-400"><span className="material-symbols-outlined">calculate</span></div>
                <div>
                  <span className="block text-xs uppercase font-black text-slate-500">Reward Calculation</span>
                  <span className="text-sm font-bold">Tính toán tổng Reward dựa trên II, AMP, Tox...</span>
                </div>
              </div>

              <div className="flex justify-center"><span className="material-symbols-outlined text-slate-600">sync</span></div>

              <div className="flex items-center gap-4 bg-emerald-500/20 p-4 rounded-2xl border border-emerald-500/30 ring-2 ring-emerald-500/20 animate-pulse">
                <div className="bg-emerald-500 p-2 rounded-lg text-white"><span className="material-symbols-outlined">trending_up</span></div>
                <div>
                  <span className="block text-xs uppercase font-black text-emerald-400">Gradient Update</span>
                  <span className="text-sm font-bold">Cập nhật trọng số theo Self-Critical Objective</span>
                </div>
              </div>
            </div>
          </div>

          {/* Cơ chế Thưởng Phạt */}
          <div className="flex flex-col gap-6 justify-center">
            <div className="bg-white dark:bg-slate-800/50 p-6 rounded-3xl border border-blue-100 dark:border-blue-900 shadow-sm">
              <div className="flex items-center gap-3 mb-4 text-blue-600 dark:text-blue-400">
                <span className="material-symbols-outlined text-4xl">add_circle</span>
                <h4 className="font-black text-xl uppercase tracking-tight">Hàm Thưởng</h4>
              </div>
              <p className="text-slate-600 dark:text-slate-300 font-medium mb-4 text-sm leading-relaxed">
                Thúc đẩy mô hình hội tụ về các vùng không gian có đặc tính sinh học tốt vượt xa dữ liệu huấn luyện.
              </p>
              <ul className="space-y-2">
                <li className="flex items-center gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-xl border border-blue-100 dark:border-blue-800 text-sm font-bold text-blue-800 dark:text-blue-200">
                  <span className="bullet text-xs">●</span> Độ ổn định nhiệt động (II &lt; 40)
                </li>
                <li className="flex items-center gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-xl border border-blue-100 dark:border-blue-800 text-sm font-bold text-blue-800 dark:text-blue-200">
                  <span className="bullet text-xs">●</span> Khả năng kháng khuẩn (Prob. AMP)
                </li>
              </ul>
            </div>

            <div className="bg-white dark:bg-slate-800/50 p-6 rounded-3xl border border-rose-100 dark:border-rose-900 shadow-sm">
              <div className="flex items-center gap-3 mb-4 text-rose-600 dark:text-rose-400">
                <span className="material-symbols-outlined text-4xl">remove_circle</span>
                <h4 className="font-black text-xl uppercase tracking-tight">Hàm Phạt</h4>
              </div>
              <p className="text-slate-600 dark:text-slate-300 font-medium mb-4 text-sm leading-relaxed">
                Mô hình phạt nặng các chuỗi peptide có nguy cơ gây độc hoặc không đủ tính đa dạng.
              </p>
              <ul className="space-y-2">
                <li className="flex items-center gap-2 bg-rose-50 dark:bg-rose-900/20 p-3 rounded-xl border border-rose-100 dark:border-rose-900 text-sm font-bold text-rose-800 dark:text-rose-200">
                  <span className="bullet text-xs">●</span> Độc tính tế bào
                </li>
                <li className="flex items-center gap-2 bg-rose-50 dark:bg-rose-900/20 p-3 rounded-xl border border-rose-100 dark:border-rose-900 text-sm font-bold text-rose-800 dark:text-rose-200">
                  <span className="bullet text-xs">●</span> Vi phạm các ràng buộc hóa lý
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 11,
    title: "KẾT QUẢ THỰC NGHIỆM",
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
    id: 12,
    title: "SO SÁNH KẾT QUẢ",
    content: (
      <div className="flex flex-col h-full space-y-6">
        <div className="overflow-x-auto bg-white/40 dark:bg-slate-900/40 rounded-3xl p-6 border border-slate-200 dark:border-slate-700">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b-2 border-slate-200 dark:border-slate-700">
                <th className="py-4 px-4 font-black uppercase text-slate-500 text-sm">Mô hình (Model)</th>
                <th className="py-4 px-4 font-black uppercase text-blue-600 text-sm">Stable Ratio (%)</th>
                <th className="py-4 px-4 font-black uppercase text-emerald-600 text-sm">Prob. AMP (%)</th>
                <th className="py-4 px-4 font-black uppercase text-purple-600 text-sm">Uniqueness (%)</th>
              </tr>
            </thead>
            <tbody className="font-medium text-lg">
              <tr className="border-b border-slate-100 dark:border-slate-800 hover:bg-white/20">
                <td className="py-4 px-4 font-bold">HydrAMP</td>
                <td className="py-4 px-4">58.42%</td>
                <td className="py-4 px-4">68.15%</td>
                <td className="py-4 px-4 text-emerald-500">100%</td>
              </tr>
              <tr className="border-b border-slate-100 dark:border-slate-800 hover:bg-white/20">
                <td className="py-4 px-4 font-bold">ESM2-Decoder</td>
                <td className="py-4 px-4">42.18%</td>
                <td className="py-4 px-4">62.40%</td>
                <td className="py-4 px-4">98.24%</td>
              </tr>
              <tr className="border-b border-slate-100 dark:border-slate-800 hover:bg-white/20">
                <td className="py-4 px-4 font-bold">PepGraphormer</td>
                <td className="py-4 px-4">65.80%</td>
                <td className="py-4 px-4 font-bold text-emerald-500">75.12%</td>
                <td className="py-4 px-4">99.10%</td>
              </tr>
              <tr className="bg-blue-500/10 text-blue-700 dark:text-blue-300 font-black">
                <td className="py-5 px-4 animate-pulse">LPG (Ours)</td>
                <td className="py-5 px-4 text-xl">71.24%</td>
                <td className="py-5 px-4">73.58%</td>
                <td className="py-5 px-4 text-emerald-500">100%</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="grid md:grid-cols-2 gap-6 flex-1">
          <div className="bg-emerald-50/50 dark:bg-emerald-900/10 p-6 rounded-2xl border border-emerald-100 dark:border-emerald-800 flex items-center gap-4">
            <span className="material-symbols-outlined text-4xl text-emerald-500">verified</span>
            <p className="text-lg font-medium"><strong>Độ ổn định vượt trội:</strong> LPG đạt 71.24%, cao nhất trong các mô hình so sánh nhờ GATv2 và phần thưởng RL cho SCST.</p>
          </div>
          <div className="bg-blue-50/50 dark:bg-blue-900/10 p-6 rounded-2xl border border-blue-100 dark:border-blue-800 flex items-center gap-4">
            <span className="material-symbols-outlined text-4xl text-blue-500">rocket_launch</span>
            <p className="text-lg font-medium"><strong>Hiệu năng cân bằng:</strong> LPG duy trì tính mới  và hoạt tính cao trong khi đảm bảo cấu trúc bền vững nhất.</p>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 13,
    title: "CÁC VẤN ĐỀ CẦN CẢI THIỆN",
    content: (
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 h-full items-center">
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[300px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">analytics</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Đánh giá In-silico</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Kết quả dựa trên tính toán mô phỏng, cần đối chiếu với thực nghiệm Wet-lab (MIC assays) để xác nhận hoạt tính chính xác.</p>
        </div>
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[300px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">hub</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Động lực học Phân tử</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">GATv2 hiện chỉ mô phỏng trạng thái tĩnh, chưa tính đến sự biến đổi cấu trúc peptide theo thời gian trong môi trường dung môi.</p>
        </div>
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[300px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">straighten</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Giới hạn Độ dài</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Mô hình tối ưu cho peptide ngắn (&lt;50 aa), cần cải thiện khả năng sinh tạo các chuỗi protein dài và phức tạp hơn.</p>
        </div>
        <div className="p-8 bg-slate-100 dark:bg-slate-800 text-center rounded-3xl border border-slate-200 dark:border-slate-700 h-[300px] flex flex-col items-center justify-center hover:shadow-lg transition">
          <span className="material-symbols-outlined text-5xl text-slate-400 mb-4">settings_input_component</span>
          <h4 className="font-bold text-slate-800 dark:text-slate-100 text-lg mb-2">Đa mục tiêu đồng thời</h4>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Cần tích hợp các bộ lọc độc tính và độ tan (Solubility) sâu hơn để tối ưu hóa đồng thời nhiều chỉ số sinh học.</p>
        </div>
      </div>
    )
  },
  {
    id: 14,
    title: "ĐÓNG GÓP VÀ Ý NGHĨA",
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
    id: 15,
    title: "ĐỊNH HƯỚNG PHÁT TRIỂN",
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
    title: "DEMO",
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
