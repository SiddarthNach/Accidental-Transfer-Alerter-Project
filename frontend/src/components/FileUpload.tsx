import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClearFile: () => void;
  isUploading: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  selectedFile,
  onClearFile,
  isUploading
}) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    multiple: false,
    disabled: isUploading
  });

  if (selectedFile) {
    return (
      <Card className="p-6 border-2 border-dashed border-primary/20 bg-secondary/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <FileText className="h-8 w-8 text-primary" />
            <div>
              <p className="font-medium text-foreground">{selectedFile.name}</p>
              <p className="text-sm text-muted-foreground">
                {(selectedFile.size / 1024).toFixed(1)} KB
              </p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onClearFile}
            disabled={isUploading}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <Card
      {...getRootProps()}
      className={`p-8 border-2 border-dashed transition-all duration-200 cursor-pointer ${
        isDragActive
          ? 'border-primary bg-primary/5 scale-105'
          : 'border-muted-foreground/25 hover:border-primary/50 hover:bg-secondary/30'
      }`}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center justify-center space-y-4 text-center">
        <div className={`p-4 rounded-full ${
          isDragActive ? 'bg-primary/10' : 'bg-secondary'
        } transition-colors duration-200`}>
          <Upload className={`h-8 w-8 ${
            isDragActive ? 'text-primary' : 'text-muted-foreground'
          }`} />
        </div>
        <div>
          <p className="text-lg font-medium mb-1">
            {isDragActive ? 'Drop your CSV file here' : 'Upload CSV File'}
          </p>
          <p className="text-sm text-muted-foreground">
            Drag and drop your transaction data or click to browse
          </p>
        </div>
        <Button variant="outline" size="sm" className="mt-2">
          Browse Files
        </Button>
      </div>
    </Card>
  );
};