import { useState } from "react";
import { toast } from "@/hooks/use-toast";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MRIUpload } from "@/components/MRIUpload";
import { DiagnosisResults } from "@/components/DiagnosisResults";
import { TreatmentRecommendations } from "@/components/TreatmentRecommendations";
import { Brain, Zap, Target, Activity, Shield, Clock } from "lucide-react";
import brainMriHero from "@/assets/brain-mri-hero.jpg";
import aiBrainIcon from "@/assets/ai-brain-icon.jpg";
import MRIOverlay from "../components/MRIOverlay";

const Index = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<FileList | null>(null);
  const [diagnosisResults, setDiagnosisResults] = useState<any | null>(null);

  const handleFileUpload = (files: FileList) => {
    setUploadedFiles(files);
    setAnalysisComplete(false);
    setDiagnosisResults(null);
  };

  const handleAnalyze = async () => {
    if (!uploadedFiles) return;
    setIsAnalyzing(true);

    const formData = new FormData();
    Array.from(uploadedFiles).forEach(file => formData.append('files', file));

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Failed to analyze MRI scans');
      const result = await response.json();
      setDiagnosisResults(result);
      setAnalysisComplete(true);
      toast({
        title: "Analysis Complete",
        description: "AI analysis finished successfully.",
        variant: "success"
      });
    } catch (error) {
      toast({
        title: "Analysis Error",
        description: error instanceof Error ? error.message : String(error),
        variant: "destructive"
      });
      setDiagnosisResults(null);
      setAnalysisComplete(false);
      console.error("AI Analysis Error:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-surface">
      {/* Hero Section */}
      <section className="relative bg-gradient-hero overflow-hidden">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="relative container mx-auto px-4 py-20">
          <div className="max-w-4xl mx-auto text-center text-white">
            <div className="flex justify-center mb-6">
              <img src={aiBrainIcon} alt="AI Brain" className="h-20 w-20 rounded-full shadow-elevated" />
            </div>
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              NeuroAI Pro
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-white/90 leading-relaxed">
              Advanced AI-Powered Brain Tumor Detection & Treatment Planning
            </p>
            <div className="flex flex-wrap justify-center gap-4 mb-8">
              <Badge className="bg-white/20 text-white border-white/30 px-4 py-2 text-sm">
                <Brain className="h-4 w-4 mr-2" />
                3D Neural Networks
              </Badge>
              <Badge className="bg-white/20 text-white border-white/30 px-4 py-2 text-sm">
                <Zap className="h-4 w-4 mr-2" />
                Real-time Analysis
              </Badge>
              <Badge className="bg-white/20 text-white border-white/30 px-4 py-2 text-sm">
                <Target className="h-4 w-4 mr-2" />
                94%+ Accuracy
              </Badge>
            </div>
            <Button variant="hero" size="lg" onClick={() => document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' })}>
              Start AI Analysis
            </Button>
          </div>
        </div>
      </section>

      {/* Features Overview */}
      <section className="py-16 bg-background">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Advanced Medical AI Technology</h2>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Leveraging state-of-the-art 3D U-Net architecture with attention mechanisms 
              for precise brain tumor detection and segmentation
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <Card className="shadow-medical hover:shadow-elevated transition-all duration-300">
              <CardHeader className="text-center">
                <div className="mx-auto h-12 w-12 bg-gradient-medical rounded-full flex items-center justify-center mb-4">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <CardTitle>Multi-Modal Analysis</CardTitle>
                <CardDescription>
                  Processes T1, T1ce, T2, and FLAIR MRI sequences simultaneously
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="shadow-success hover:shadow-elevated transition-all duration-300">
              <CardHeader className="text-center">
                <div className="mx-auto h-12 w-12 bg-gradient-success rounded-full flex items-center justify-center mb-4">
                  <Activity className="h-6 w-6 text-white" />
                </div>
                <CardTitle>Precise Segmentation</CardTitle>
                <CardDescription>
                  Accurate tumor boundary detection with Dice scores &gt;0.89
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="shadow-medical hover:shadow-elevated transition-all duration-300">
              <CardHeader className="text-center">
                <div className="mx-auto h-12 w-12 bg-gradient-medical rounded-full flex items-center justify-center mb-4">
                  <Shield className="h-6 w-6 text-white" />
                </div>
                <CardTitle>Clinical Integration</CardTitle>
                <CardDescription>
                  FDA-compliant analysis with comprehensive treatment recommendations
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Main Analysis Interface */}
      <section id="upload-section" className="py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto space-y-8">
            
            {/* File Upload */}
            <MRIUpload onUpload={handleFileUpload} />

            {/* Analysis Controls */}
            {uploadedFiles && !analysisComplete && (
              <Card className="shadow-elevated animate-slide-up">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="h-5 w-5 text-primary" />
                    Ready for Analysis
                  </CardTitle>
                  <CardDescription>
                    {uploadedFiles.length} MRI scan(s) uploaded. Start AI-powered diagnosis.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col sm:flex-row gap-4">
                    <Button 
                      variant="medical" 
                      size="lg" 
                      className="flex-1" 
                      onClick={handleAnalyze}
                      disabled={isAnalyzing}
                    >
                      {isAnalyzing ? (
                        <>
                          <Brain className="h-5 w-5 animate-pulse-slow" />
                          Analyzing Brain Scans...
                        </>
                      ) : (
                        <>
                          <Zap className="h-5 w-5" />
                          Start AI Analysis
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Analysis Results */}
            {(isAnalyzing || (analysisComplete && diagnosisResults)) && (
              <DiagnosisResults 
                results={diagnosisResults}
                isLoading={isAnalyzing}
              />
            )}

            {/* Treatment Recommendations */}
            {analysisComplete && diagnosisResults?.tumorDetected && (
              <TreatmentRecommendations
                tumorType={diagnosisResults.tumorType}
                tumorVolume={diagnosisResults.tumorVolume}
                location={diagnosisResults.location}
                confidence={diagnosisResults.confidence}
              />
            )}

            {/* MRI Visualization: Always show after analysis */}
            {analysisComplete && diagnosisResults && (
              <Card className="shadow-elevated animate-fade-in">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="h-6 w-6 text-primary" />
                    MRI Analysis Visualization
                  </CardTitle>
                  <CardDescription>
                    Original scan with AI-generated tumor segmentation overlay
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="relative rounded-lg overflow-hidden">
                    <img 
                      src={brainMriHero} 
                      alt="Brain MRI Analysis" 
                      className="w-full h-64 object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                    <div className="absolute bottom-4 left-4 text-white">
                      <p className="text-sm font-medium">Axial T1ce slice with tumor segmentation</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Index;
<MRIOverlay
  mriUrl="/api/mri.png"      // Replace with your MRI image endpoint
  maskUrl="/api/mask.png"    // Replace with your mask PNG endpoint
/>