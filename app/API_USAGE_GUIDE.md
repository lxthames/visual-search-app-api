# Visual Search API - Usage Guide

## How to Import Insomnia Collection

1. **Open Insomnia** application
2. Click on **"Create"** → **"Import/Export"** → **"Import Data"**
3. Select **"From File"**
4. Choose the `insomnia_collection.json` file
5. The collection will appear in your workspace as "Visual Search API"

## API Endpoints

Base URL: `http://localhost:8000`

### 1. Health Check
**GET** `/api/health`

Check if the API server is running.

**Response:**
```json
{
  "status": "ok"
}
```

---

### 2. Index Shelf
**POST** `/api/index-shelf`

Upload a shelf image to detect and index objects. This endpoint uses GroundingDINO to detect objects based on a text prompt, then indexes them in the vector database.

**Parameters (Form Data):**
- `file` (file, required): Image file (PNG, JPG, etc.)
- `prompt` (text, optional): Text prompt for object detection (default: "Bottle")
  - Examples: "Bottle", "Can", "Product", "Box", "Container"
- `box_thresh` (text, optional): Detection threshold between 0.0 and 1.0 (default: 0.3)
  - Lower values = fewer but more confident detections
  - Higher values = more detections but may include false positives

**Example Request:**
- Select an image file in the `file` field
- Set `prompt` to "Bottle" (or whatever objects you want to detect)
- Set `box_thresh` to 0.3

**Response:**
```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "num_detections": 15,
  "num_indexed": 14
}
```

**Important:** Save the `image_id` - you'll need it for searching!

---

### 3. List Shelves
**GET** `/api/shelves`

Get a list of all indexed shelves with the number of objects in each.

**Response:**
```json
[
  {
    "shelf_id": "550e8400-e29b-41d4-a716-446655440000",
    "num_objects": 14
  },
  {
    "shelf_id": "660e8400-e29b-41d4-a716-446655440001",
    "num_objects": 8
  }
]
```

---

### 4. Search Similar (JSON)
**POST** `/api/search`

Search for visually similar objects on a specific shelf. Returns JSON with bounding boxes and similarity scores.

**Parameters (Form Data):**
- `file` (file, required): Query image to search for
- `shelf_id` (text, required): The ID of the shelf to search in (from Index Shelf or List Shelves)
- `max_results` (text, optional): Maximum number of results (default: 10)
- `match_threshold` (text, optional): Maximum distance threshold (default: 0.19)
  - Lower values = only very similar matches
  - Higher values = more matches but less similar
  - Distance metric: smaller = more similar

**Example Request:**
- Select a query image in the `file` field
- Set `shelf_id` to a shelf ID from List Shelves
- Set `max_results` to 10
- Set `match_threshold` to 0.19

**Response:**
```json
{
  "matches": [
    {
      "bbox": [100, 50, 200, 150],
      "score": 0.12
    },
    {
      "bbox": [300, 100, 400, 200],
      "score": 0.15
    }
  ]
}
```

**Bounding Box Format:** `[x1, y1, x2, y2]` (top-left and bottom-right coordinates)

---

### 5. Search Similar (Visual)
**POST** `/api/search-visual`

Same as Search Similar, but returns a PNG image with matches highlighted on the shelf image.

**Parameters:** Same as Search Similar (JSON)

**Response:** PNG image with bounding boxes:
- **GREEN boxes** = matches (score < threshold)
- **RED boxes** = non-matches (score >= threshold)
- **Score** displayed above each box

**Usage in Insomnia:**
- The response will be a PNG image
- You can right-click and "Save Response" to download the image
- Or view it directly in Insomnia's response preview

---

## Step-by-Step Workflow

### 1. Start the Server
```bash
cd "C:\Users\umarb\Downloads\visual semtic\app"
python main.py
```

Wait for: `Server will be available at http://localhost:8000`

### 2. Index a Shelf Image
1. Use **"Index Shelf"** endpoint in Insomnia
2. Upload a shelf image (e.g., a photo of products on a shelf)
3. Set prompt to what you want to detect (e.g., "Bottle", "Can")
4. Send request
5. **Copy the `image_id` from the response**

### 3. List All Shelves (Optional)
1. Use **"List Shelves"** endpoint
2. See all indexed shelves and their object counts

### 4. Search for Similar Objects
1. Use **"Search Similar (JSON)"** or **"Search Similar (Visual)"**
2. Upload a query image (e.g., a single product you want to find)
3. Paste the `shelf_id` from step 2
4. Adjust `match_threshold` if needed (start with 0.19)
5. Send request
6. View results:
   - JSON: List of bounding boxes and scores
   - Visual: Image with highlighted matches

---

## Tips & Best Practices

### Object Detection Prompts
- Be specific: "Bottle" works better than "Object"
- Use singular form: "Bottle" not "Bottles"
- Examples that work well:
  - "Bottle"
  - "Can"
  - "Box"
  - "Container"
  - "Product"

### Threshold Settings
- **box_thresh** (for indexing):
  - 0.2-0.3: Good for clean images
  - 0.4-0.5: For cluttered images or when you want more detections
  
- **match_threshold** (for searching):
  - 0.15-0.19: Strict matching (only very similar)
  - 0.20-0.25: Moderate matching
  - 0.25+: Loose matching (more results)

### Image Requirements
- Supported formats: PNG, JPG, JPEG
- Higher resolution = better detection
- Clear, well-lit images work best
- Avoid blurry or very dark images

### Performance Notes
- First request may be slow (loading AI models)
- Subsequent requests are faster
- Large images take longer to process

---

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use
- Make sure all dependencies are installed
- Check error messages in the console

### No detections found
- Try lowering `box_thresh` (e.g., 0.2)
- Try a different prompt
- Check if the image is clear enough

### No search results
- Verify the `shelf_id` is correct
- Try increasing `match_threshold` (e.g., 0.25)
- Make sure you indexed the shelf first

### Import errors in Insomnia
- Make sure you're importing the JSON file, not a text file
- Try restarting Insomnia
- Check that the file is valid JSON

---

## Alternative: Using cURL

If you prefer command line:

```bash
# Health check
curl http://localhost:8000/api/health

# Index shelf
curl -X POST "http://localhost:8000/api/index-shelf?prompt=Bottle&box_thresh=0.3" \
  -F "file=@shelf_image.jpg"

# List shelves
curl http://localhost:8000/api/shelves

# Search (JSON)
curl -X POST "http://localhost:8000/api/search?shelf_id=YOUR_SHELF_ID&max_results=10&match_threshold=0.19" \
  -F "file=@query_image.jpg" \
  -o results.json

# Search (Visual)
curl -X POST "http://localhost:8000/api/search-visual?shelf_id=YOUR_SHELF_ID&max_results=10&match_threshold=0.19" \
  -F "file=@query_image.jpg" \
  -o result_image.png
```

---

## API Documentation

Interactive API documentation is available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide interactive testing and detailed endpoint documentation.

