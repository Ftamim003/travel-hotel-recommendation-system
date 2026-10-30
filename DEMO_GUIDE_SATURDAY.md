# TravelBD Saturday Demo Guide

## 1. Run the project

Open this folder in VS Code, then run these commands in the terminal:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

If you have already created the virtual environment, activate it and run `pip install -r requirements.txt`, then the last two commands.

## 2. Demo flow

1. On the home page, choose a budget (for example BDT 5,000 per person per day), 3 days, 2 travelers, and enter interests such as `beach, nature, photography`.
2. Show the destination shortlist and the explanation for each ranking.
3. Open **Cox's Bazar Sea Beach** (destination ID `D018`) to demonstrate official BPC hotels. The results should include Hotel Shaibal, Motel Labonee, Motel Upal and Motel Probal where present in the supplied BPC file.
4. Point out that a listing can have an official source link while its price or facilities are unavailable. Do not present missing values as zero-cost rooms.
5. On a covered destination such as Cox's Bazar, Dhaka, Sylhet, Bandarban, Rangamati, Rajshahi, Chattogram or Moulvibazar, show the BanTABSA location-level review context. Explain that this affects destination ranking slightly but is not an individual named hotel's rating.
6. From the destination detail page, click **Plan my trip**. Select 3 days, a daily attraction budget, travel style (for example Nature), and interests such as `beach, nature`. Generate the itinerary and explain the Morning/Afternoon/Evening slots, estimated costs and source labels.
7. Open **Model and Data** to explain the method and data limitations.

## 3. Important points to say during the presentation

- Destination matching uses TF-IDF content similarity; a transparent hybrid score also considers budget, supplied review aggregates, season, origin, and a small BanTABSA location-level signal.
- The 24 official BPC listings are included in the hotel recommendation table and matching official listings are surfaced for the selected destination.
- BanTABSA contains 1,242 Bengali reviews across 31 anonymized hotel IDs and 9 locations. It is used only as location-level context because its anonymized IDs cannot be reliably linked to named BPC listings.
- Many of the original 10,000 trip/review records and 463 hotel rows are synthetic placeholders. They are demonstration data, not verified real customer behavior.
- The itinerary uses 1,573 generated attraction records across 78 destinations, built from the destination highlight lists and tourism geodata; some geospatial associations are approximate.
- Itinerary entry costs and durations are heuristics for demonstration, not verified fees or schedules. The planner does not verify opening hours, traffic, local transport time, or weather.
- Prices, transport fares, and travel times should be described as supplied estimates or captured listing information, not live quotes. The system does not book hotels or check live availability.

## 4. Before the demo

- Run `python train_model.py` after every dataset/code update.
- Check that the homepage and recommendation form load.
- Try at least one destination with official BPC listings (Cox's Bazar is a good example).
- Test the itinerary planner with both Cox's Bazar and a destination with fewer mapped attractions (for example Old Dhaka); the number of stops can differ by available data.
- Do not delete the `data` or `artifacts` folders; the app needs them.
