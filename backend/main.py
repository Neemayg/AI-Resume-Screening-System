"""
AI Resume Screening System - FastAPI Backend
Main entry point for the application.

This system:
1. Accepts job descriptions (text or file)
2. Accepts exactly 5 resume files (PDF/DOCX)
3. Extracts and preprocesses text
4. Computes similarity using TF-IDF and cosine similarity
5. Ranks candidates and returns structured results
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Local imports
from config import (
    CORS_ORIGINS, API_TITLE, API_DESCRIPTION, API_VERSION,
    JD_UPLOAD_DIR, RESUME_UPLOAD_DIR, MAX_RESUMES, ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB
)
from models import (
    JobDescriptionText, ResumeResult, AnalysisResponse,
    UploadResponse, ErrorResponse, HealthResponse, CompareRequest,
    FitCategory
)
from resume_parser import resume_parser
from ai_service import ai_service
from nlp_engine import nlp_engine

# V2 System Imports
from utils.data_loader import data_loader
from matching.text_normalizer import text_normalizer
from matching.skill_extractor import skill_extractor
from matching.role_detector import role_detector
from matching.section_parser import section_parser
from matching.skill_matcher import skill_matcher
from matching.semantic_scorer import semantic_scorer
from ranking.ranking_engine import RankingEngine
from dataclasses import asdict

# Initialize V2 Engine
v2_ranking_engine = RankingEngine(
    skill_normalizer=skill_extractor,
    role_detector=role_detector,
    skill_matcher=skill_matcher,
    experience_analyzer=None, # Integrated logic
    semantic_analyzer=semantic_scorer,
    penalty_calculator=None,
    job_roles_data=data_loader._job_roles,
    skills_data=data_loader._skills_dataset
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Application State ---
class AppState:
    """Global application state management."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset application state."""
        self.jd_text: Optional[str] = None
        self.jd_preprocessed: Optional[str] = None
        self.jd_analysis: Optional[dict] = None
        self.resumes: List[dict] = []
        self.results: Optional[List[ResumeResult]] = None
        self.analysis_complete: bool = False

app_state = AppState()


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting AI Resume Screening System...")
    
    # Verify NLP engine is ready
    if not nlp_engine.is_ready():
        logger.warning("NLP Engine datasets not fully loaded")
    else:
        logger.info("NLP Engine ready with all datasets")
    
    # Clear old uploads
    for directory in [JD_UPLOAD_DIR, RESUME_UPLOAD_DIR]:
        if directory.exists():
            for file in directory.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {file}: {e}")
    
    logger.info("System ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Resume Screening System...")
    app_state.reset()


# --- FastAPI App ---
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# CORS Middleware - Hardcoded for debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow ALL origins
    allow_credentials=True, # Allow credentials (cookies/auth)
    allow_methods=["*"], # Allow ALL methods (POST, GET, etc)
    allow_headers=["*"], # Allow ALL headers
)


# --- Helper Functions ---
def validate_file_extension(filename: str) -> bool:
    """Validate file has allowed extension."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def generate_file_id() -> str:
    """Generate unique file ID."""
    return str(uuid.uuid4())[:8]


def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination."""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# --- API Endpoints ---

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload_jd": "/upload-jd",
            "upload_jd_text": "/upload-jd-text",
            "upload_resumes": "/upload-resumes",
            "analyze": "/analyze",
            "results": "/results",
            "reset": "/reset"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        nlp_ready=nlp_engine.is_ready(),
        datasets_loaded=bool(nlp_engine.skills_data and nlp_engine.job_roles)
    )


@app.post("/upload-jd", response_model=UploadResponse, tags=["Upload"])
def upload_job_description_file(file: UploadFile = File(...)):
    """
    Upload a job description file (PDF, DOCX, or TXT).
    
    The file content will be extracted and stored for analysis.
    """
    logger.info(f"Received JD file upload: {file.filename}")
    
    # Validate file extension
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique filename
    file_id = generate_file_id()
    ext = Path(file.filename).suffix.lower()
    new_filename = f"jd_{file_id}{ext}"
    file_path = JD_UPLOAD_DIR / new_filename
    
    try:
        # Save file
        save_upload_file(file, file_path)
        logger.info(f"JD file saved: {file_path}")
        
        # Extract text
        extracted_text, success = resume_parser.extract_text(str(file_path))
        
        if not success or not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text from the uploaded file"
            )
        
        # Preprocess text
        preprocessed = nlp_engine.preprocess(extracted_text)
        
        # Store in state
        app_state.jd_text = extracted_text
        app_state.jd_preprocessed = preprocessed
        app_state.results = None  # Reset results
        app_state.analysis_complete = False
        
        logger.info(f"JD processed successfully: {len(extracted_text)} chars")
        
        return UploadResponse(
            success=True,
            message="Job description uploaded and processed successfully",
            filename=file.filename,
            file_id=file_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing JD file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/upload-jd-text", response_model=UploadResponse, tags=["Upload"])
def upload_job_description_text(jd: JobDescriptionText):
    """
    Upload job description as plain text.
    
    Minimum 50 characters required.
    """
    logger.info(f"Received JD text upload: {len(jd.text)} chars")
    
    if len(jd.text.strip()) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description must be at least 50 characters"
        )
    
    try:
        # Preprocess text
        preprocessed = nlp_engine.preprocess(jd.text)
        
        # Store in state
        app_state.jd_text = jd.text
        app_state.jd_preprocessed = preprocessed
        app_state.results = None
        app_state.analysis_complete = False
        
        logger.info("JD text processed successfully")
        
        return UploadResponse(
            success=True,
            message="Job description text received and processed",
            filename="text_input",
            file_id=generate_file_id()
        )
        
    except Exception as e:
        logger.error(f"Error processing JD text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )


@app.post("/upload-resumes", tags=["Upload"])
def upload_resumes(files: List[UploadFile] = File(...)):
    """
    Upload resume files (PDF or DOCX).
    
    Exactly 5 resumes are required for analysis.
    Files can be uploaded in multiple batches until 5 are received.
    """
    logger.info(f"Received {len(files)} resume file(s)")
    
    # Check if adding these would exceed limit
    current_count = len(app_state.resumes)
    if current_count + len(files) > MAX_RESUMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot add {len(files)} resumes. Current: {current_count}, Maximum: {MAX_RESUMES}"
        )
    
    uploaded = []
    errors = []
    
    for file in files:
        # Validate extension
        if not validate_file_extension(file.filename):
            errors.append(f"{file.filename}: Invalid file type")
            continue
        
        # Generate unique ID and filename
        file_id = generate_file_id()
        ext = Path(file.filename).suffix.lower()
        new_filename = f"resume_{file_id}{ext}"
        file_path = RESUME_UPLOAD_DIR / new_filename
        
        try:
            # Save file
            save_upload_file(file, file_path)
            
            # Extract text
            extracted_text, success = resume_parser.extract_text(str(file_path))
            
            if not success or not extracted_text.strip():
                errors.append(f"{file.filename}: Could not extract text")
                file_path.unlink()  # Remove failed file
                continue
            
            # Validate content
            is_valid, validation_msg = resume_parser.validate_resume_content(extracted_text)
            if not is_valid:
                errors.append(f"{file.filename}: {validation_msg}")
                file_path.unlink()
                continue
            
            # Preprocess
            preprocessed = nlp_engine.preprocess(extracted_text)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Store resume data
            resume_data = {
                "id": file_id,
                "name": file.filename,
                "size": format_file_size(file_size),
                "path": str(file_path),
                "original_text": extracted_text,
                "preprocessed_text": preprocessed,
                "status": "ready"
            }
            
            app_state.resumes.append(resume_data)
            uploaded.append({
                "id": file_id,
                "name": file.filename,
                "size": format_file_size(file_size),
                "status": "ready"
            })
            
            logger.info(f"Resume processed: {file.filename}")
            
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
            logger.error(f"Error processing resume {file.filename}: {str(e)}")
    
    # Reset results when resumes change
    app_state.results = None
    app_state.analysis_complete = False
    
    return {
        "success": len(uploaded) > 0,
        "message": f"Uploaded {len(uploaded)} resume(s). Total: {len(app_state.resumes)}/{MAX_RESUMES}",
        "uploaded": uploaded,
        "errors": errors if errors else None,
        "total_resumes": len(app_state.resumes),
        "required": MAX_RESUMES,
        "ready_for_analysis": len(app_state.resumes) == MAX_RESUMES
    }


@app.delete("/remove-resume/{resume_id}", tags=["Upload"])
async def remove_resume(resume_id: str):
    """Remove a specific resume by ID."""
    
    for i, resume in enumerate(app_state.resumes):
        if resume["id"] == resume_id:
            # Remove file
            try:
                Path(resume["path"]).unlink()
            except:
                pass
            
            # Remove from state
            app_state.resumes.pop(i)
            app_state.results = None
            app_state.analysis_complete = False
            
            return {
                "success": True,
                "message": f"Resume {resume['name']} removed",
                "total_resumes": len(app_state.resumes)
            }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Resume with ID {resume_id} not found"
    )


@app.get("/resumes", tags=["Upload"])
async def get_uploaded_resumes():
    """Get list of currently uploaded resumes."""
    return {
        "resumes": [
            {
                "id": r["id"],
                "name": r["name"],
                "size": r["size"],
                "status": r["status"]
            }
            for r in app_state.resumes
        ],
        "total": len(app_state.resumes),
        "required": MAX_RESUMES,
        "ready_for_analysis": len(app_state.resumes) == MAX_RESUMES and app_state.jd_text is not None
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze_resumes():
    """
    Analyze all uploaded resumes against the job description using V2 Advanced Scoring.
    """
    logger.info("Starting resume analysis (V2 System)...")
    
    # Validate prerequisites
    if not app_state.jd_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No job description uploaded. Please upload a JD first."
        )
    
    if len(app_state.resumes) != MAX_RESUMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exactly {MAX_RESUMES} resumes required. Current: {len(app_state.resumes)}"
        )
    
    try:
        # 1. Analyze Job Description
        jd_analysis = v2_ranking_engine.analyze_job_description(app_state.jd_text)
        app_state.jd_analysis = jd_analysis # Store for /results endpoint
        
        logger.info(f"V2 Analysis - Role: {jd_analysis.get('detected_role_id')}")

        # 2. Prepare Resumes (Parse sections)
        resume_data_list = []
        for resume in app_state.resumes:
            try:
                # Parse resume into structured sections
                sections = section_parser.parse_resume(resume["original_text"])
                
                # Convert to dict for engine
                resume_dict = asdict(sections)
                resume_dict['id'] = resume['id']
                resume_dict['name'] = resume['name']
                resume_dict['text'] = resume['original_text']
                # Ensure extracted skills list is present effectively
                resume_dict['extracted_skills'] = sections.skills 
                
                resume_data_list.append(resume_dict)
            except Exception as e:
                logger.error(f"Error parsing resume {resume['name']}: {e}")
                # Fallback to basic data
                resume_data_list.append({
                    'id': resume['id'],
                    'name': resume['name'],
                    'text': resume['original_text'],
                    'extracted_skills': [],
                    'experience': [],
                    'education': []
                })

        # 3. Run Ranking Engine
        ranked_resumes = v2_ranking_engine.rank_resumes(
            resume_data_list, 
            app_state.jd_text,
            parallel=False # Sequential for safety in web req
        )

        logger.info(f"Analysis complete. Top score: {ranked_resumes[0].final_score if ranked_resumes else 0}")
        
        # 4. Map Results to API Model
        api_results = []
        for r in ranked_resumes:
            # Map Category to FitCategory
            cat = r.category
            if "Excellent" in cat or "Strong" in cat:
                fit = FitCategory.HIGH
            elif "Good" in cat:
                fit = FitCategory.MEDIUM
            else:
                fit = FitCategory.LOW
            
            # Construct explanation/breakdown
            explanation = {
                "strengths": r.strengths,
                "weaknesses": r.weaknesses,
                "score_breakdown": r.score_breakdown
            }
            
            summary_text = f"{r.recommendation}. "
            if r.strengths:
                summary_text += f"Key strengths: {', '.join(r.strengths[:2])}. "
            if r.weaknesses:
                summary_text += f"Areas for review: {', '.join(r.weaknesses[:2])}."

            api_results.append(ResumeResult(
                rank=r.rank,
                id=r.resume_id,
                resume_name=r.candidate_name,
                match_score=int(round(r.final_score)),
                fit=fit,
                matched_skills=r.matched_skills,
                missing_skills=r.missing_critical_skills,
                summary=summary_text,
                skill_breakdown=r.score_breakdown['component_scores'],
                explanation=explanation
            ))
            
        # Store results
        app_state.results = api_results
        app_state.analysis_complete = True
        
        # Get simplified statistics
        stats = {
            "average_score": sum(r.match_score for r in api_results) / len(api_results) if api_results else 0,
            "top_score": api_results[0].match_score if api_results else 0,
            "distribution": {
                "High": sum(1 for r in api_results if r.fit == FitCategory.HIGH),
                "Medium": sum(1 for r in api_results if r.fit == FitCategory.MEDIUM),
                "Low": sum(1 for r in api_results if r.fit == FitCategory.LOW)
            }
        }

        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully (V2 Engine)",
            job_title_detected=jd_analysis.get('detected_role', {}).get('detected_role_title'),
            total_candidates=len(api_results),
            results=api_results,
            analysis_metadata={
                "jd_skills_count": len(jd_analysis.get('all_required_skills', [])),
                "detected_role_confidence": jd_analysis.get('detected_role', {}).get('confidence'),
                "statistics": stats
            }
        )

    except Exception as e:
        logger.error(f"V2 Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/results", tags=["Analysis"])
def get_results(use_ai: bool = False):
    """
    Get the latest analysis results.
    
    Set use_ai=true for AI-enhanced summaries (slower but more detailed).
    Default is fast mode without AI.
    """
    if not app_state.analysis_complete or not app_state.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis results available. Please run /analyze first."
        )
    
    # Calculate statistics from stored ResumeResult objects
    scores = [r.match_score for r in app_state.results]
    stats = {
        "average_score": sum(scores) / len(scores) if scores else 0,
        "top_score": max(scores) if scores else 0,
        "lowest_score": min(scores) if scores else 0,
        "high_fit_count": sum(1 for r in app_state.results if r.fit == FitCategory.HIGH),
        "medium_fit_count": sum(1 for r in app_state.results if r.fit == FitCategory.MEDIUM),
        "low_fit_count": sum(1 for r in app_state.results if r.fit == FitCategory.LOW),
    }
    
    # Fast mode - return results as-is
    if not use_ai:
        return {
            "success": True,
            "job_title_detected": app_state.jd_analysis.get("detected_role", {}).get("detected_role_title") if app_state.jd_analysis else None,
            "total_candidates": len(app_state.results),
            "results": [
                {
                    "rank": r.rank,
                    "id": r.id,
                    "resume_name": r.resume_name,
                    "match_score": r.match_score,
                    "fit": r.fit.value,
                    "matched_skills": r.matched_skills,
                    "missing_skills": r.missing_skills,
                    "summary": r.summary,
                    "skill_breakdown": r.skill_breakdown,
                    "explanation": r.explanation
                }
                for r in app_state.results
            ],
            "statistics": stats,
            "ai_enhanced": False
        }
    
    # AI mode - generate enhanced summaries
    enhanced_results = []
    for r in app_state.results:
        resume_data = next((res for res in app_state.resumes if res["id"] == r.id), None)
        ai_summary = r.summary
        ai_skills = r.matched_skills
        
        if resume_data and app_state.jd_text:
            try:
                ai_result = ai_service.analyze_resume(
                    resume_data["original_text"],
                    app_state.jd_text
                )
                if ai_result.get("summary"):
                    ai_summary = ai_result["summary"]
                if ai_result.get("skills_found"):
                    ai_skills = ai_result["skills_found"]
            except Exception as e:
                logger.warning(f"AI analysis failed for {r.resume_name}: {e}")
        
        enhanced_results.append({
            "rank": r.rank,
            "id": r.id,
            "resume_name": r.resume_name,
            "match_score": r.match_score,
            "fit": r.fit.value,
            "matched_skills": ai_skills,
            "missing_skills": r.missing_skills,
            "summary": ai_summary
        })
    
    return {
        "success": True,
        "job_title_detected": app_state.jd_analysis.get("detected_role") if app_state.jd_analysis else None,
        "total_candidates": len(app_state.results),
        "results": enhanced_results,
        "statistics": stats,
        "ai_enhanced": True
    }


@app.post("/reset", tags=["System"])
async def reset_system():
    """
    Reset the system state.
    
    Clears all uploaded files and analysis results.
    """
    logger.info("Resetting system state...")
    
    # Clear uploaded files
    for directory in [JD_UPLOAD_DIR, RESUME_UPLOAD_DIR]:
        if directory.exists():
            for file in directory.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {file}: {e}")
    
    # Reset state
    app_state.reset()
    
    logger.info("System reset complete")
    
    return {
        "success": True,
        "message": "System reset successfully. All data cleared."
    }


@app.get("/jd-status", tags=["Upload"])
async def get_jd_status():
    """Get current job description status and analysis."""
    if not app_state.jd_text:
        return {
            "uploaded": False,
            "message": "No job description uploaded"
        }
    
    return {
        "uploaded": True,
        "text_length": len(app_state.jd_text),
        "analysis": app_state.jd_analysis
    }





@app.post("/improve-jd", tags=["Analysis"])
async def improve_job_description(jd: JobDescriptionText):
    """
    Analyze and suggest improvements for the job description.
    """
    if len(jd.text.strip()) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description must be at least 50 characters"
        )
        
    try:
        result = ai_service.improve_job_description(jd.text)
        return {
            "success": True,
            "analysis": result
        }
    except Exception as e:
        logger.error(f"JD Improvement error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/compare", tags=["Analysis"])
async def compare_candidates(request: CompareRequest):
    """
    Compare selected candidates side-by-side.
    
    Requires:
    - Job description to be uploaded
    - Analysis to be completed
    - At least 2 candidate IDs
    """
    # Validate prerequisites
    if not app_state.jd_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description required. Please upload a JD first."
        )
    
    if not app_state.analysis_complete or not app_state.results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis must be completed before comparison. Please run analysis first."
        )
        
    if len(request.candidate_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Select at least 2 candidates to compare"
        )
    
    # Extract candidate data for comparison
    candidates_to_compare = []
    for candidate_id in request.candidate_ids:
        # Find the resume data
        resume_data = next((r for r in app_state.resumes if r["id"] == candidate_id), None)
        
        if not resume_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Candidate {candidate_id} not found"
            )
        
        # Get the analysis result for additional context
        result_data = next((r for r in app_state.results if r.id == candidate_id), None)
        
        # Prepare candidate data for AI service
        candidate_info = {
            "name": resume_data["name"],
            "original_text": resume_data["original_text"],
            "matched_skills": result_data.matched_skills if result_data else [],
            "match_score": result_data.match_score if result_data else 0
        }
        candidates_to_compare.append(candidate_info)
    
    if len(candidates_to_compare) < 2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find data for selected candidates"
        )
    
    try:
        # Call AI service to compare candidates
        comparison_result = ai_service.compare_candidates(candidates_to_compare, app_state.jd_text)
        
        logger.info(f"Comparison completed for {len(candidates_to_compare)} candidates")
        
        return {
            "success": True,
            "comparison": comparison_result
        }
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )



# --- Error Handlers ---
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )