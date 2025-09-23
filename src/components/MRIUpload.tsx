import { useState, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, FileText, Brain, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface MRIUploadProps {
  onUpload: (files: FileList) => void;
}

export const MRIUpload = ({ onUpload }: MRIUploadProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const { toast } = useToast();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      processFiles(files);
    }
  }, []);

  const processFiles = (files: File[]) => {
    const validFiles = files.filter(file => 
      file.name.endsWith('.nii') || 
      file.name.endsWith('.nii.gz') || 
      file.name.endsWith('.dcm')
    );

    if (validFiles.length === 0) {
      toast({
        title: "Invalid file format",
        description: "Please upload NIfTI (.nii, .nii.gz) or DICOM (.dcm) files",
        variant: "destructive"
      });
      return;
    }

    setUploadedFiles(validFiles);
    const fileList = new DataTransfer();
    validFiles.forEach(file => fileList.items.add(file));
    onUpload(fileList.files);

    toast({
      title: "Files uploaded successfully",
      description: `${validFiles.length} MRI scan(s) ready for analysis`,
    });
  };

  return (
    <Card className="shadow-elevated bg-gradient-surface">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <Brain className="h-6 w-6 text-primary" />
          Upload MRI Scans
        </CardTitle>
        <CardDescription>
          Upload brain MRI scans in NIfTI or DICOM format for AI-powered tumor detection
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
            isDragOver
              ? "border-primary bg-accent/20 shadow-medical"
              : "border-border hover:border-primary/50"
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            multiple
            accept=".nii,.nii.gz,.dcm"
            onChange={handleFileSelect}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          
          <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">Drop MRI files here</h3>
          <p className="text-muted-foreground mb-4">
            or click to select files
          </p>
          
          <div className="flex flex-wrap gap-2 justify-center mb-4">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs bg-primary/10 text-primary">
              .nii
            </span>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs bg-primary/10 text-primary">
              .nii.gz
            </span>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs bg-primary/10 text-primary">
              .dcm
            </span>
          </div>

          <div className="flex items-center gap-2 text-sm text-muted-foreground justify-center">
            <AlertCircle className="h-4 w-4" />
            <span>Expected: T1, T1ce, T2, FLAIR modalities</span>
          </div>
        </div>

        {uploadedFiles.length > 0 && (
          <div className="mt-6">
            <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Uploaded Files ({uploadedFiles.length})
            </h4>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {uploadedFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-2 bg-accent/50 rounded-md text-sm"
                >
                  <span className="font-mono">{file.name}</span>
                  <span className="text-muted-foreground">
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};