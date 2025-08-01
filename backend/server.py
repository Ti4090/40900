from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime, timedelta
import uuid
from emergentintegrations.llm.chat import LlmChat, UserMessage
import json
import re
from typing import Optional, List

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(MONGO_URL)
db = client.vocabulary_app
words_collection = db.words
history_collection = db.search_history
favorites_collection = db.favorites
notes_collection = db.word_notes
stats_collection = db.user_stats
quiz_collection = db.quiz_results

# Gemini API key
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', "AIzaSyBCif8yqRojbnmJK1NyS4VQyKJ4ng0xsKY")

class WordRequest(BaseModel):
    word: str

class FavoriteRequest(BaseModel):
    word: str
    word_data: dict

class NoteRequest(BaseModel):
    word: str
    note: str

class QuizAnswer(BaseModel):
    word: str
    user_answer: str
    correct_answer: str
    is_correct: bool

async def analyze_word_with_gemini(word: str) -> dict:
    """Analyze word using Gemini AI with structured prompting"""
    
    system_message = """Sen İngilizce öğrenen Türkler için kelime öğretmeni bir asistansın. Verilen İngilizce kelimeyi kapsamlı şekilde analiz et ve aşağıdaki JSON formatında yanıt ver:

{
  "word": "kelime",
  "turkish_translation": "türkçe karşılığı",
  "pronunciation": "IPA fonetic gösterim",
  "english_explanation": "kelimenin ingilizce açıklaması",
  "turkish_explanation": "kelimenin türkçe açıklaması",
  "difficulty_level": "Beginner/Intermediate/Advanced",
  "synonyms": ["eş anlamlı kelimeler listesi"],
  "antonyms": ["zıt anlamlı kelimeler listesi"],
  "example_sentences": [
    {
      "english": "İngilizce örnek cümle",
      "turkish": "Türkçe çevirisi"
    }
  ],
  "important_notes": "Önemli notlar ve kullanım ipuçları",
  "phrasal_verbs": [
    {
      "phrase": "phrasal verb",
      "meaning": "anlamı",
      "example": "örnek cümle"
    }
  ],
  "word_family": ["kelime ailesi - run, running, runner gibi"],
  "common_mistakes": "Sık yapılan hatalar ve dikkat edilmesi gerekenler"
}

Sadece JSON formatında yanıt ver, başka açıklama ekleme."""

    try:
        chat = LlmChat(
            api_key=GEMINI_API_KEY,
            session_id=f"word_analysis_{uuid.uuid4().hex[:8]}",
            system_message=system_message
        ).with_model("gemini", "gemini-2.0-flash").with_max_tokens(4000)
        
        user_message = UserMessage(text=f"Bu kelimeyi analiz et: {word}")
        response = await chat.send_message(user_message)
        
        # Clean response and parse JSON
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI analiz hatası: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Vocabulary learning API is running"}

@app.post("/api/analyze-word")
async def analyze_word(request: WordRequest):
    """Analyze a word and return comprehensive information"""
    
    word = request.word.strip().lower()
    
    if not word:
        raise HTTPException(status_code=400, detail="Kelime boş olamaz")
    
    if not re.match(r'^[a-zA-Z\s\-\']+$', word):
        raise HTTPException(status_code=400, detail="Sadece İngilizce karakterler kullanın")
    
    try:
        # Check if word already exists in database
        existing_word = await words_collection.find_one({"word": word})
        
        if existing_word:
            word_data = existing_word
        else:
            # Analyze with Gemini AI
            analysis = await analyze_word_with_gemini(word)
            
            # Save to database
            word_data = {
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow(),
                **analysis
            }
            await words_collection.insert_one(word_data)
        
        # Save to search history
        history_entry = {
            "id": str(uuid.uuid4()),
            "word": word,
            "searched_at": datetime.utcnow()
        }
        await history_collection.insert_one(history_entry)
        
        # Update daily stats
        today = datetime.utcnow().date()
        await stats_collection.update_one(
            {"date": today.isoformat()},
            {
                "$inc": {"words_searched": 1},
                "$addToSet": {"unique_words": word},
                "$set": {"last_activity": datetime.utcnow()}
            },
            upsert=True
        )
        
        # Remove MongoDB _id field for response
        if "_id" in word_data:
            del word_data["_id"]
        
        return word_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Word analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Kelime analizi sırasında hata: {str(e)}")

@app.post("/api/favorites/add")
async def add_to_favorites(request: FavoriteRequest):
    """Add word to favorites"""
    try:
        favorite_entry = {
            "id": str(uuid.uuid4()),
            "word": request.word,
            "word_data": request.word_data,
            "added_at": datetime.utcnow()
        }
        
        # Check if already exists
        existing = await favorites_collection.find_one({"word": request.word})
        if existing:
            raise HTTPException(status_code=400, detail="Kelime zaten favorilerde")
        
        await favorites_collection.insert_one(favorite_entry)
        return {"message": "Kelime favorilere eklendi", "word": request.word}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Favorilere eklenirken hata: {str(e)}")

@app.delete("/api/favorites/{word}")
async def remove_from_favorites(word: str):
    """Remove word from favorites"""
    try:
        result = await favorites_collection.delete_one({"word": word})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Kelime favorilerde bulunamadı")
        return {"message": "Kelime favorilerden çıkarıldı", "word": word}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Favorilerden çıkarılırken hata: {str(e)}")

@app.get("/api/favorites")
async def get_favorites():
    """Get user's favorite words"""
    try:
        cursor = favorites_collection.find().sort("added_at", -1)
        favorites = []
        async for doc in cursor:
            if "_id" in doc:
                del doc["_id"]
            favorites.append(doc)
        return {"favorites": favorites}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Favoriler yüklenemedi: {str(e)}")

@app.post("/api/notes")
async def save_word_note(request: NoteRequest):
    """Save a note for a word"""
    try:
        note_entry = {
            "id": str(uuid.uuid4()),
            "word": request.word,
            "note": request.note,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Update if exists, create if not
        await notes_collection.update_one(
            {"word": request.word},
            {"$set": note_entry},
            upsert=True
        )
        
        return {"message": "Not kaydedildi", "word": request.word}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Not kaydedilirken hata: {str(e)}")

@app.get("/api/notes/{word}")
async def get_word_note(word: str):
    """Get note for a specific word"""
    try:
        note = await notes_collection.find_one({"word": word})
        if not note:
            return {"note": "", "word": word}
        
        if "_id" in note:
            del note["_id"]
        return note
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Not yüklenemedi: {str(e)}")

@app.get("/api/stats")
async def get_learning_stats():
    """Get learning statistics"""
    try:
        # Get last 7 days stats
        week_ago = datetime.utcnow().date() - timedelta(days=7)
        cursor = stats_collection.find({
            "date": {"$gte": week_ago.isoformat()}
        }).sort("date", 1)
        
        daily_stats = []
        total_words = 0
        total_unique = set()
        current_streak = 0
        
        async for doc in cursor:
            if "_id" in doc:
                del doc["_id"]
            daily_stats.append(doc)
            total_words += doc.get("words_searched", 0)
            total_unique.update(doc.get("unique_words", []))
        
        # Calculate streak
        today = datetime.utcnow().date()
        for i in range(30):  # Check last 30 days
            check_date = today - timedelta(days=i)
            stat = await stats_collection.find_one({"date": check_date.isoformat()})
            if stat and stat.get("words_searched", 0) > 0:
                current_streak += 1
            else:
                break
        
        # Get total favorites count
        favorites_count = await favorites_collection.count_documents({})
        
        return {
            "daily_stats": daily_stats,
            "total_words_searched": total_words,
            "unique_words_learned": len(total_unique),
            "current_streak": current_streak,
            "total_favorites": favorites_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İstatistikler yüklenemedi: {str(e)}")

@app.get("/api/quiz/generate")
async def generate_quiz():
    """Generate a quiz from recent words"""
    try:
        # Get 10 recent words for quiz
        cursor = history_collection.find().sort("searched_at", -1).limit(10)
        recent_words = []
        async for doc in cursor:
            word_data = await words_collection.find_one({"word": doc["word"]})
            if word_data:
                recent_words.append(word_data)
        
        if len(recent_words) < 3:
            raise HTTPException(status_code=400, detail="Quiz için yeterli kelime yok. Daha fazla kelime arayın.")
        
        # Generate quiz questions
        quiz_questions = []
        for i, word_data in enumerate(recent_words[:5]):  # Take 5 words for quiz
            question = {
                "id": i + 1,
                "word": word_data["word"],
                "question": f"'{word_data['word']}' kelimesinin Türkçe karşılığı nedir?",
                "correct_answer": word_data["turkish_translation"],
                "options": [word_data["turkish_translation"]]
            }
            
            # Add distractors from other words
            for other_word in recent_words:
                if other_word["word"] != word_data["word"] and len(question["options"]) < 4:
                    question["options"].append(other_word["turkish_translation"])
            
            # Shuffle options
            import random
            random.shuffle(question["options"])
            quiz_questions.append(question)
        
        return {"quiz_questions": quiz_questions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz oluşturulamadı: {str(e)}")

@app.post("/api/quiz/submit")
async def submit_quiz_results(answers: List[QuizAnswer]):
    """Submit quiz results"""
    try:
        correct_count = sum(1 for answer in answers if answer.is_correct)
        total_count = len(answers)
        score_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        quiz_result = {
            "id": str(uuid.uuid4()),
            "completed_at": datetime.utcnow(),
            "total_questions": total_count,
            "correct_answers": correct_count,
            "score_percentage": score_percentage,
            "answers": [answer.dict() for answer in answers]
        }
        
        await quiz_collection.insert_one(quiz_result)
        
        # Update daily stats
        today = datetime.utcnow().date()
        await stats_collection.update_one(
            {"date": today.isoformat()},
            {
                "$inc": {"quizzes_completed": 1},
                "$set": {"last_activity": datetime.utcnow()}
            },
            upsert=True
        )
        
        return {
            "message": "Quiz tamamlandı!",
            "score": f"{correct_count}/{total_count}",
            "percentage": score_percentage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz sonuçları kaydedilemedi: {str(e)}")

@app.get("/api/search-history")
async def get_search_history(limit: int = 20):
    """Get recent search history"""
    try:
        cursor = history_collection.find().sort("searched_at", -1).limit(limit)
        history = []
        async for doc in cursor:
            if "_id" in doc:
                del doc["_id"]
            history.append(doc)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geçmiş yüklenemedi: {str(e)}")

@app.get("/api/popular-words")
async def get_popular_words(limit: int = 10):
    """Get most searched words"""
    try:
        pipeline = [
            {"$group": {"_id": "$word", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        
        popular = []
        async for doc in history_collection.aggregate(pipeline):
            popular.append({
                "word": doc["_id"],
                "search_count": doc["count"]
            })
        
        return {"popular_words": popular}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Popüler kelimeler yüklenemedi: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)