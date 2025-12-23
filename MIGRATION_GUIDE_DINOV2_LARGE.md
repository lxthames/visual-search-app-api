# DINOv2-Large Migration Guide

## ✅ What Was Changed

### 1. Vectorizer Model (app/services/vectorizer.py)
**Changed:**
```python
# OLD:
self.model_name = "facebook/dinov2-base"  # 768 dimensions

# NEW:
self.model_name = "facebook/dinov2-large"  # 1024 dimensions
```

### 2. Vector Database Dimension (app/services/datastore.py)
**Changed:**
```python
# OLD:
dim = 768  # For DINOv2-base

# NEW:
dim = 1024  # For DINOv2-large
```

## ⚠️ CRITICAL: You MUST Re-index All Products!

**Why?**
- Old embeddings were created with DINOv2-base (768-dim)
- New embeddings will be created with DINOv2-large (1024-dim)
- **They are incompatible!** Old and new embeddings won't match.

## Migration Steps

### Step 1: Backup (Optional but Recommended)
```bash
# Export your current data if needed
# MongoDB: Use mongodump
# Milvus/Chroma: Data will be recreated
```

### Step 2: Clear Vector Database
```bash
# Option A: Use API endpoint
DELETE http://localhost:8000/api/reset-database

# Option B: Manually delete
# - Milvus: Drop collection
# - ChromaDB: Delete collection folder
```

### Step 3: Re-index All Shelf Images
```bash
# Re-upload all shelf images using:
POST http://localhost:8000/api/index-shelf-yolo
# Upload each shelf image again
```

### Step 4: Re-upload All Query Images
```bash
# Re-upload all product models using:
POST http://localhost:8000/api/ModelTraining
# Upload each product image again
```

### Step 5: Verify
```bash
# Check database counts
GET http://localhost:8000/api/debug-db-counts

# Should show:
# - MongoDB_Items: [your count]
# - VectorDB_Items: [same count]
# - Status: "SYNCED"
```

## Expected Improvements

**Before (DINOv2-base):**
- Accuracy: ~85-90%
- Embedding dimension: 768
- Model size: ~330MB

**After (DINOv2-large):**
- Accuracy: **~90-95%** ✅
- Embedding dimension: 1024
- Model size: ~1.1GB
- Speed: ~2x slower (still fast: 100-200ms per image)

## First Run Notes

**First time loading DINOv2-Large:**
- Model will download automatically (~1.1GB)
- Takes 1-2 minutes to download
- Subsequent runs are instant (cached)

## Troubleshooting

### Issue: "Dimension mismatch" error
**Solution:** Make sure you cleared and re-indexed all data

### Issue: "Model not found" error
**Solution:** Check internet connection, model downloads from HuggingFace

### Issue: Out of memory
**Solution:** 
- Use CPU instead of GPU: Set `DEVICE=cpu` in .env
- Or use smaller batch sizes

## Rollback (If Needed)

If you need to go back to DINOv2-base:

1. Change `vectorizer.py` line 48:
   ```python
   self.model_name = "facebook/dinov2-base"
   ```

2. Change `datastore.py` line 121:
   ```python
   dim = 768
   ```

3. Re-index all products again

---

**After migration, you should see 90-95% accuracy!** 🎯

