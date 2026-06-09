import React, { useState, useMemo, useRef } from 'react';

const ImageStatus = {
  Pending: 'pending',
  Processing: 'processing',
  Completed: 'completed',
  Error: 'error'
};

export default function LungAIApp() {
  const [images, setImages] = useState([]);
  const [currentFilter, setCurrentFilter] = useState('all');
  const [viewMode, setViewMode] = useState('list');
  const [selectedImage, setSelectedImage] = useState(null);
  const [isProcessingAll, setIsProcessingAll] = useState(false);

  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  const stats = useMemo(() => {
    return {
      total: images.length,
      accepted: images.filter(img => img.tag === 'yes').length,
      rejected: images.filter(img => img.tag === 'no').length,
    };
  }, [images]);

  const filteredImages = useMemo(() => {
    if (currentFilter === 'all') return images;
    return images.filter(img => img.tag === currentFilter);
  }, [images, currentFilter]);

  const handleFilesAdded = (fileList) => {
    const newImages = [];
    Array.from(fileList).forEach(file => {
      const ext = file.name.split('.').pop().toLowerCase();
      if (['jpg', 'jpeg', 'png', 'bmp', 'dcm'].includes(ext)) {
        newImages.push({
          id: crypto.randomUUID(),
          name: file.name,
          file: file, 
          fileUrl: URL.createObjectURL(file),
          prepUrl: null, 
          tag: null, 
          status: ImageStatus.Pending,
        });
      }
    });

    if (newImages.length > 0) {
      setImages(prev => [...prev, ...newImages]);
      setCurrentFilter('all');
    }
    
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (folderInputRef.current) folderInputRef.current.value = '';
  };

  const onDropFiles = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files?.length > 0) {
      handleFilesAdded(e.dataTransfer.files);
    }
  };

  const processSingleImage = async (imageObj) => {
    setImages(prev => prev.map(img => 
      img.id === imageObj.id ? { ...img, status: ImageStatus.Processing } : img
    ));

    const formData = new FormData();
    formData.append("file", imageObj.file);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/preprocess", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("서버 응답 오류");

      const blob = await response.blob();
      const prepUrl = URL.createObjectURL(blob);

      setImages(prev => prev.map(img => 
        img.id === imageObj.id 
          ? { ...img, prepUrl, status: ImageStatus.Completed, tag: 'yes' } 
          : img
      ));

      if (selectedImage && selectedImage.id === imageObj.id) {
        setSelectedImage(prev => ({ ...prev, prepUrl, status: ImageStatus.Completed, tag: 'yes' }));
      }

    } catch (error) {
      console.error("API 통신 에러:", error);
      setImages(prev => prev.map(img => 
        img.id === imageObj.id 
          ? { ...img, status: ImageStatus.Error, tag: 'no' } 
          : img
      ));
    }
  };

  const onRunClassification = async () => {
    const pendingImages = images.filter(img => img.status === ImageStatus.Pending);
    if (pendingImages.length === 0) {
      alert("판별 대기 중인 이미지가 없습니다.");
      return;
    }

    setIsProcessingAll(true);
    for (const img of pendingImages) {
      await processSingleImage(img);
    }
    setIsProcessingAll(false);
    alert("모든 이미지의 전처리가 완료되었습니다!");
  };

  const onFileSelected = async (image) => {
    setSelectedImage(image);
    setViewMode('detail');

    if (image.name.toLowerCase().endsWith('.dcm') && !image.dicomPreviewUrl) {
      const formData = new FormData();
      formData.append("file", image.file);

      try {
        const response = await fetch("http://127.0.0.1:8000/api/preview", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const blob = await response.blob();
          const previewUrl = URL.createObjectURL(blob);
          
          setSelectedImage(prev => prev?.id === image.id ? { ...prev, dicomPreviewUrl: previewUrl } : prev);
          setImages(prev => prev.map(img => img.id === image.id ? { ...img, dicomPreviewUrl: previewUrl } : img));
        }
      } catch (error) {
        console.error("DICOM 미리보기 로드 실패:", error);
      }
    }
  };

  const onBackToList = () => {
    setSelectedImage(null);
    setViewMode('list');
  };

  return (
    <div className="flex h-screen w-full bg-[#FAFAFA] font-sans text-[#030213]">
      
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={(e) => handleFilesAdded(e.target.files)} 
        multiple 
        accept="image/png, image/jpeg, image/bmp, .dcm" 
        className="hidden" 
      />

      <input 
        type="file" 
        ref={folderInputRef} 
        onChange={(e) => handleFilesAdded(e.target.files)} 
        webkitdirectory="true" 
        directory="true" 
        multiple 
        className="hidden" 
      />

      <div className="w-[320px] bg-white border-r border-[#E0E0E0] flex flex-col flex-shrink-0">
        <div className="p-6 border-b border-[#E0E0E0]">
          <h1 className="text-2xl font-medium text-[#030213]">Lung-Ai Pro</h1>
          <p className="text-sm text-[#717182] mt-1">SOTA 전처리 품질 선별 시스템</p>
        </div>

        <div className="p-6 overflow-y-auto flex-1 flex flex-col gap-6">
          <div className="flex flex-col">
            <h2 className="text-lg font-medium text-[#030213] mb-4">AI Processing</h2>
            
            <button 
              onClick={onRunClassification} 
              disabled={isProcessingAll}
              className={`flex items-center justify-center py-3 px-4 rounded-lg font-medium transition-opacity mb-4 
                ${isProcessingAll ? 'bg-gray-400 cursor-not-allowed text-white' : 'bg-[#030213] hover:bg-black/90 text-white'}`}
            >
              {isProcessingAll ? "AI 서버 처리 중..." : "AI 품질 판별 실행"}
            </button>

            <div className="flex flex-col gap-2 mt-4">
              <button onClick={() => setCurrentFilter('all')} className={`flex items-center justify-between w-full border py-3 px-3 rounded-lg transition-colors ${currentFilter === 'all' ? 'bg-[#ECECF0] border-[#A0A0A0]' : 'bg-white border-[#E0E0E0]'}`}>
                <div className="flex items-center">
                  <div className="w-4 h-4 bg-[#717182] rounded-full mr-2"></div><span className="font-medium">Total</span>
                </div>
                <span className="font-medium">{stats.total}</span>
              </button>

              <button onClick={() => setCurrentFilter('yes')} className={`flex items-center justify-between w-full border py-3 px-3 rounded-lg transition-colors ${currentFilter === 'yes' ? 'bg-[#ECECF0] border-[#A0A0A0]' : 'bg-white border-[#E0E0E0]'}`}>
                <div className="flex items-center">
                  <div className="w-4 h-4 bg-[#22C55E] rounded-full mr-2"></div><span className="font-medium">Accepted</span>
                </div>
                <span className="font-medium">{stats.accepted}</span>
              </button>

              <button onClick={() => setCurrentFilter('no')} className={`flex items-center justify-between w-full border py-3 px-3 rounded-lg transition-colors ${currentFilter === 'no' ? 'bg-[#ECECF0] border-[#A0A0A0]' : 'bg-white border-[#E0E0E0]'}`}>
                <div className="flex items-center">
                  <div className="w-4 h-4 bg-[#EF4444] rounded-full mr-2"></div><span className="font-medium">Rejected</span>
                </div>
                <span className="font-medium">{stats.rejected}</span>
              </button>
            </div>
          </div>

          <div className="flex flex-col mt-4">
            <h2 className="text-lg font-medium text-[#030213] mb-4">Folder Management</h2>
            <button onClick={() => folderInputRef.current?.click()} className="bg-white border border-[#E0E0E0] hover:bg-[#ECECF0] h-16 rounded-lg font-medium mb-2">
              이미지 폴더 불러오기
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-[1400px] flex flex-col gap-6">
          
          {viewMode === 'list' && (
            <div>
              <h2 className="text-lg font-medium text-[#030213] mb-4">Image Upload</h2>
              <div 
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }}
                onDrop={onDropFiles}
                className="border-2 border-dashed border-[#E0E0E0] hover:border-[#030213] bg-white transition-colors rounded-lg p-12 flex flex-col items-center justify-center cursor-pointer"
              >
                <span className="text-base font-medium">개별 파일 업로드</span>
                <span className="text-sm text-[#717182] mt-2">클릭하여 파일들을 선택하거나 이곳에 드롭하세요</span>
              </div>
            </div>
          )}

          <div>
            <h2 className="text-lg font-medium text-[#030213] mb-4">Image Review</h2>
            <div className="bg-white border border-[#E0E0E0] rounded-lg p-6 min-h-[500px]">
              
              {viewMode === 'list' && (
                <div className="flex flex-col gap-2">
                  {filteredImages.map((file) => (
                    <div key={file.id} onClick={() => onFileSelected(file)} className="flex items-center justify-between p-3 px-4 bg-[#FAFAFA] border border-[#E0E0E0] hover:bg-[#ECECF0] hover:border-[#030213] rounded-md cursor-pointer transition-all">
                      <span className="text-sm font-medium truncate pr-4">{file.name}</span>
                      
                      {file.status === ImageStatus.Pending && <div className="bg-gray-200 text-gray-600 text-xs px-3 py-1 rounded">대기중</div>}
                      {file.status === ImageStatus.Processing && <div className="bg-blue-100 text-blue-600 text-xs px-3 py-1 rounded animate-pulse">AI 분석중...</div>}
                      {file.tag === 'yes' && <div className="bg-green-500 text-white text-xs px-3 py-1 rounded">✓ Accepted (완료)</div>}
                      {file.tag === 'no' && <div className="bg-red-500 text-white text-xs px-3 py-1 rounded">× Error (오류)</div>}
                    </div>
                  ))}
                  {filteredImages.length === 0 && <p className="text-center text-gray-400 py-10">데이터가 없습니다.</p>}
                </div>
              )}

              {viewMode === 'detail' && selectedImage && (
                <div className="animate-in fade-in duration-200">
                  <button onClick={onBackToList} className="text-sm text-[#717182] hover:text-black mb-4">← 목록으로 돌아가기</button>
                  <h3 className="text-base font-medium mb-4">{selectedImage.name}</h3>
                  
                  <div className="grid grid-cols-2 gap-6">
                    
                    {/* 💡 1. 좌측: 원본 X-ray 뷰어 (DICOM 썸네일 지원) */}
                    <div className="flex flex-col">
                      <span className="font-medium mb-3">원본 X-ray</span>
                      <div className="bg-black rounded-lg h-[500px] flex items-center justify-center border border-gray-200 overflow-hidden">
                        {selectedImage.name.toLowerCase().endsWith('.dcm') ? (
                          selectedImage.dicomPreviewUrl ? (
                            <img src={selectedImage.dicomPreviewUrl} alt="Original DICOM" className="max-w-full max-h-full object-contain animate-in fade-in" />
                          ) : (
                            <span className="text-gray-400 text-sm animate-pulse">DICOM 렌더링 중...</span>
                          )
                        ) : (
                          <img src={selectedImage.fileUrl} alt="Original" className="max-w-full max-h-full object-contain" />
                        )}
                      </div>
                    </div>

                    {/* 💡 2. 우측: Lung-Ai 전처리 결과 뷰어 (복구됨!) */}
                    <div className="flex flex-col">
                      <span className="font-medium mb-3">Lung-Ai 전처리 결과</span>
                      <div className="bg-black rounded-lg h-[500px] flex items-center justify-center border border-gray-200 overflow-hidden relative">
                        {selectedImage.status === ImageStatus.Processing && (
                          <span className="text-white absolute z-10 animate-pulse">AI가 뼈 음영을 제거하는 중입니다...</span>
                        )}
                        {selectedImage.prepUrl ? (
                          <img src={selectedImage.prepUrl} alt="Preprocessed" className="max-w-full max-h-full object-contain" />
                        ) : (
                          <span className="text-gray-500">{selectedImage.status === ImageStatus.Pending ? 'AI 판별을 실행해주세요.' : ''}</span>
                        )}
                      </div>
                    </div>

                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}