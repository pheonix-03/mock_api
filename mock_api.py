from fastapi import FastAPI, Request, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from typing import Optional
import pandas as pd
import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
from pytz import timezone
import aiofiles
import aiohttp
import random
import numpy as np
from dotenv import load_dotenv
from threading import Lock

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure logging with rotation (save in root directory)
handler = RotatingFileHandler("mock_error.log", maxBytes=10*1024*1024, backupCount=5)  # 10 MB limit
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# Mock credentials (no KiteConnect dependency)
LOG_API_KEY = os.getenv("LOG_API_KEY")
INSTRUMENTS_CSV = "instruments.csv"
OUTPUT_TXT = "output.txt"
CSV_FILE = "new_historical_data.csv"
ACCESS_TOKEN_FILE = "mock_access_token.txt"

# Global NSE holidays (mocked with static 2025 holidays)
NSE_HOLIDAYS = []

# Thread-safe manual expiry dates
manual_expiry_dates = {
    "weekly": None,
    "monthly": None
}
expiry_lock = Lock()

# Instrument data (loaded from CSV or output.txt)
INSTRUMENT_DATA = None
NIFTY_OPTION_TOKENS = []

# Cache for mock prices (instrument_token: last_price)
MOCK_PRICE_CACHE = {}
MOCK_PRICE_CACHE_LOCK = Lock()

async def fetch_nse_holidays():
    """
    Mock fetching trading holidays and store in NSE_HOLIDAYS.
    Uses a static list of 2025 holidays instead of NSE API.
    """
    global NSE_HOLIDAYS
    try:
        # Static list of 2025 NSE holidays
        mock_holidays = [
            {"tradingDate": "26-Jan-2025"},  # Republic Day
            {"tradingDate": "04-Mar-2025"},  # Mahashivratri
            {"tradingDate": "20-Mar-2025"},  # Holi
            {"tradingDate": "14-Apr-2025"},  # Dr. Ambedkar Jayanti
            {"tradingDate": "15-Aug-2025"},  # Independence Day
            {"tradingDate": "02-Oct-2025"},  # Gandhi Jayanti
            {"tradingDate": "09-Nov-2025"},  # Diwali
        ]
        NSE_HOLIDAYS = [
            datetime.datetime.strptime(item["tradingDate"], "%d-%b-%Y").date()
            for item in mock_holidays
        ]
        logging.info(f"Mocked fetch of {len(NSE_HOLIDAYS)} NSE trading holidays")
    except Exception as e:
        logging.error(f"Error mocking NSE holidays: {str(e)}")
        NSE_HOLIDAYS = []

async def load_instrument_data():
    """
    Load instrument data from instruments.csv or output.txt and extract NIFTY option tokens.
    """
    global INSTRUMENT_DATA, NIFTY_OPTION_TOKENS
    try:
        if os.path.exists(INSTRUMENTS_CSV):
            INSTRUMENT_DATA = pd.read_csv(INSTRUMENTS_CSV)
            logging.info(f"Loaded instrument data from {INSTRUMENTS_CSV}")
        elif os.path.exists(OUTPUT_TXT):
            INSTRUMENT_DATA = pd.read_csv(OUTPUT_TXT)
            logging.info(f"Loaded instrument data from {OUTPUT_TXT}")
        else:
            logging.error(f"No instrument data source found ({INSTRUMENTS_CSV} or {OUTPUT_TXT})")
            raise Exception("No instrument data source found")

        # Extract NIFTY option tokens
        nifty_options = INSTRUMENT_DATA[
            (INSTRUMENT_DATA["segment"] == "NFO-OPT") &
            (INSTRUMENT_DATA["name"] == "NIFTY")
        ]
        NIFTY_OPTION_TOKENS = nifty_options["instrument_token"].tolist()
        logging.info(f"Extracted {len(NIFTY_OPTION_TOKENS)} NIFTY option tokens")
    except Exception as e:
        logging.error(f"Error loading instrument data: {str(e)}")
        INSTRUMENT_DATA = pd.DataFrame()
        raise

async def fetch_instruments():
    """
    Mock fetching instrument data and save as CSV.
    Loads from local instruments.csv or generates expanded mock data if not present.
    """
    try:
        global INSTRUMENT_DATA, NIFTY_OPTION_TOKENS
        if os.path.exists(INSTRUMENTS_CSV):
            INSTRUMENT_DATA = pd.read_csv(INSTRUMENTS_CSV)
            logging.info(f"Mocked instrument fetch: Loaded existing {INSTRUMENTS_CSV}")
        else:
            # Generate expanded mock instrument data
            mock_data = [
                {
                    "instrument_token": 256265,
                    "exchange": "NSE",
                    "tradingsymbol": "NIFTY 50",
                    "name": "NIFTY 50",
                    "segment": "NSE",
                    "instrument_type": "INDEX"
                }
            ]
            # Add NIFTY options for multiple strikes and expiries
            strikes = [24500, 24750, 25000, 25250, 25500]
            expiries = ["2025-05-22", "2025-05-29", "2025-06-05"]
            token = 12345
            for expiry in expiries:
                for strike in strikes:
                    for opt_type in ["CE", "PE"]:
                        mock_data.append({
                            "instrument_token": token,
                            "exchange": "NFO",
                            "tradingsymbol": f"NIFTY25{expiry[5:7]}{expiry[8:10]}{strike}{opt_type}",
                            "name": "NIFTY",
                            "segment": "NFO-OPT",
                            "instrument_type": opt_type,
                            "strike": strike,
                            "expiry": expiry
                        })
                        token += 1

            INSTRUMENT_DATA = pd.DataFrame(mock_data)
            async with aiofiles.open(INSTRUMENTS_CSV, "w") as f:
                await f.write(INSTRUMENT_DATA.to_csv(index=False))
            logging.info(f"Mocked instrument fetch: Generated and saved {INSTRUMENTS_CSV} with {len(mock_data)} instruments")

        # Validate required columns
        required_columns = {"instrument_token", "exchange", "tradingsymbol", "name", "segment", "instrument_type", "strike", "expiry"}
        missing = required_columns - set(INSTRUMENT_DATA.columns)
        if missing:
            logging.error(f"Missing required columns in INSTRUMENT_DATA: {missing}")
            INSTRUMENT_DATA = pd.DataFrame()
            raise Exception(f"Missing required columns: {missing}")

        nifty_options = INSTRUMENT_DATA[
            (INSTRUMENT_DATA["segment"] == "NFO-OPT") &
            (INSTRUMENT_DATA["name"] == "NIFTY")
        ]
        NIFTY_OPTION_TOKENS = nifty_options["instrument_token"].tolist()
        logging.info(f"Mocked fetch: Found {len(NIFTY_OPTION_TOKENS)} NIFTY option tokens.")
        return True
    except Exception as e:
        logging.error(f"Error mocking instruments: {str(e)}")
        return False

@app.on_event("startup")
async def startup():
    """
    Initialize mock API, fetch NSE holidays, and load/fetch instrument data.
    """
    logging.info("Starting mock API...")
    await fetch_nse_holidays()
    await fetch_instruments()
    if INSTRUMENT_DATA is None or INSTRUMENT_DATA.empty:
        try:
            await load_instrument_data()
        except Exception as e:
            logging.error(f"Startup failed to load instrument data: {str(e)}")
    if INSTRUMENT_DATA is None or INSTRUMENT_DATA.empty:
        logging.error("Failed to load instrument data on startup")
    else:
        logging.info(f"Instrument data loaded with {len(INSTRUMENT_DATA)} records")
    logging.info("Finished mock API startup.")

# API key for log endpoints
api_key_header = APIKeyHeader(name="X-API-Key")
async def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    Verify API key for protected endpoints.
    """
    if LOG_API_KEY and api_key != LOG_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.get("/debug", dependencies=[Depends(verify_api_key)])
async def debug():
    """
    Debug endpoint for mock API, inspecting instrument data and file status.
    """
    return {
        "instrument_data_loaded": INSTRUMENT_DATA is not None and not INSTRUMENT_DATA.empty,
        "nifty_option_tokens_count": len(NIFTY_OPTION_TOKENS),
        "instruments_csv_exists": os.path.exists(INSTRUMENTS_CSV),
        "output_txt_exists": os.path.exists(OUTPUT_TXT),
        "access_token_file_exists": os.path.exists(ACCESS_TOKEN_FILE),
        "mock_api": True
    }

@app.get("/callback")
async def oauth_callback(request: Request, request_token: str):
    """
    Mock OAuth callback (returns dummy token).
    """
    try:
        logging.info(f"Mock OAuth callback with request_token: {request_token[:10]}...")
        mock_access_token = "mock_access_token_1234567890"
        async with aiofiles.open(ACCESS_TOKEN_FILE, "w") as f:
            await f.write(mock_access_token)
        logging.info("Mock access token stored")
        return {"status": "success", "access_token": mock_access_token}
    except Exception as e:
        logging.error(f"/callback error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Mock OAuth callback failed: {str(e)}")

@app.get("/callback_logs", dependencies=[Depends(verify_api_key)])
async def get_callback_logs():
    """
    Return the last 10 log entries related to /callback.
    """
    log_file = "mock_error.log"
    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail="Log file not found.")
    async with aiofiles.open(log_file, "r") as f:
        lines = [line async for line in f if "/callback" in line]
    return {"callback_log": lines[-10:]}

@app.get("/get_access_token")
async def get_access_token():
    """
    Retrieve the mock access token.
    """
    if os.path.exists(ACCESS_TOKEN_FILE):
        async with aiofiles.open(ACCESS_TOKEN_FILE, "r") as f:
            access_token = await f.read()
        return {"access_token": access_token.strip()}
    else:
        return {"access_token": "mock_access_token_1234567890"}

@app.get("/fetch_historical_data")
async def fetch_last_1_minute_data():
    """
    Generate mock 1-minute NIFTY 50 data and save to CSV.
    """
    try:
        instrument_token = 256265  # NIFTY 50
        ist = timezone('Asia/Kolkata')
        end_time = datetime.datetime.now(ist).replace(second=0, microsecond=0)
        start_time = end_time - datetime.timedelta(minutes=1)

        # Generate mock data
        logging.info("Generating mock 1-minute NIFTY 50 data.")
        base_price = 25000.0  # Typical NIFTY 50 value for 2025
        fluctuation = base_price * 0.001  # ±0.1% or ±25 points
        close_price = base_price + random.uniform(-fluctuation, fluctuation)
        open_price = close_price + random.uniform(-fluctuation/2, fluctuation/2)
        high_price = max(open_price, close_price) + random.uniform(0, fluctuation/2)
        low_price = min(open_price, close_price) - random.uniform(0, fluctuation/2)
        volume = random.randint(50000, 150000)  # Realistic volume

        mock_data = [{
            "date": end_time.strftime("%Y-%m-%d %H:%M:%S+05:30"),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        }]
        df = pd.DataFrame(mock_data)
        async with aiofiles.open(CSV_FILE, "w") as f:
            await f.write(df.to_csv(index=False))
        logging.info(f"Saved mock data (1 row) to {CSV_FILE}")
        return {
            "status": "success",
            "message": f"Mock 1-minute data saved to {CSV_FILE}",
            "rows_saved": len(df),
            "mock": True
        }

    except Exception as e:
        logging.error(f"/fetch_historical_data error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating mock data: {str(e)}")

@app.get("/download_csv")
async def download_csv():
    """
    Download the mock historical data CSV file.
    """
    if not os.path.exists(CSV_FILE):
        raise HTTPException(status_code=404, detail="CSV file not found. Fetch mock data first.")
    return FileResponse(CSV_FILE, media_type="text/csv", filename=CSV_FILE)

@app.get("/health")
async def health():
    """
    Check mock API health.
    """
    return {"status": "ok", "mock": True}

def get_option_expiry(expiry_type: str = "weekly") -> datetime.date:
    """
    Calculate the next weekly or monthly expiry date, adjusted for holidays and weekends.
    """
    with expiry_lock:
        today = datetime.date.today()
        if manual_expiry_dates.get(expiry_type):
            return manual_expiry_dates[expiry_type]
        weekday = today.weekday()
        if expiry_type == "weekly":
            days_ahead = (3 - weekday + 7) % 7
            expiry = today + datetime.timedelta(days=days_ahead)
        elif expiry_type == "monthly":
            next_month = today.replace(day=28) + datetime.timedelta(days=4)
            last_day = next_month - datetime.timedelta(days=next_month.day)
            expiry = last_day
            while expiry.weekday() != 3:
                expiry -= datetime.timedelta(days=1)
        else:
            raise ValueError("expiry_type must be 'weekly' or 'monthly'")
        # Adjust for holidays and weekends
        while expiry in NSE_HOLIDAYS or expiry.weekday() > 4:
            expiry -= datetime.timedelta(days=1)
        return expiry

@app.post("/set_expiry_date")
async def set_expiry_date(expiry_type: str, expiry_date: datetime.date):
    """
    Set a manual expiry date for weekly or monthly options.
    """
    if expiry_type not in ["weekly", "monthly"]:
        raise HTTPException(status_code=400, detail="Invalid expiry type.")
    # Validate expiry date
    if expiry_date in NSE_HOLIDAYS or expiry_date.weekday() > 4:
        raise HTTPException(status_code=400, detail="Expiry date cannot be a holiday or weekend.")
    with expiry_lock:
        manual_expiry_dates[expiry_type] = expiry_date
    logging.info(f"{expiry_type.capitalize()} expiry date set to {expiry_date}")
    return {"message": f"{expiry_type.capitalize()} expiry date set to {expiry_date}."}

async def get_nifty_spot_price():
    """
    Generate mock NIFTY 50 spot price.
    """
    try:
        logging.info("Generating mock NIFTY 50 spot price.")
        with MOCK_PRICE_CACHE_LOCK:
            last_price = MOCK_PRICE_CACHE.get(256265, 25000.0)  # Default for NIFTY 50
            fluctuation = last_price * 0.001  # ±0.1%
            new_price = last_price + random.uniform(-fluctuation, fluctuation)
            new_price = max(20000.0, min(30000.0, new_price))
            MOCK_PRICE_CACHE[256265] = new_price
        return new_price
    except Exception as e:
        logging.error(f"Mock get_nifty_spot_price error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating mock LTP: {str(e)}")

@app.get("/get_price")
async def get_price(instrument_token: int):
    """
    Generate mock price for a given instrument token.
    """
    try:
        if INSTRUMENT_DATA is None or INSTRUMENT_DATA.empty:
            success = await fetch_instruments()
            if not success or INSTRUMENT_DATA is None or INSTRUMENT_DATA.empty:
                await load_instrument_data()
            if INSTRUMENT_DATA is None or INSTRUMENT_DATA.empty:
                raise HTTPException(status_code=500, detail="Instrument data not loaded: No valid data source found.")

        # Validate instrument token
        instrument_row = INSTRUMENT_DATA[INSTRUMENT_DATA["instrument_token"] == instrument_token]
        if instrument_row.empty:
            raise HTTPException(status_code=400, detail=f"Invalid instrument token: {instrument_token}")
        
        trading_symbol = instrument_row["tradingsymbol"].iloc[0]
        exchange = instrument_row["exchange"].iloc[0]
        segment = instrument_row.get("segment", "").iloc[0]
        name = instrument_row.get("name", "").iloc[0]
        instrument_type = instrument_row.get("instrument_type", "").iloc[0]

        # Generate mock price
        logging.info(f"Generating mock price for instrument token {instrument_token} ({trading_symbol}).")
        with MOCK_PRICE_CACHE_LOCK:
            last_price = MOCK_PRICE_CACHE.get(instrument_token)
            if last_price is None:
                if segment == "NFO-OPT" and name == "NIFTY":
                    base_price = 200.0  # Typical for NIFTY options
                elif segment == "NSE" and name == "NIFTY 50":
                    base_price = 25000.0  # Typical NIFTY 50 value
                else:
                    base_price = 100.0  # Default for others
                last_price = base_price

            if segment == "NFO-OPT" and name == "NIFTY":
                fluctuation = last_price * 0.05  # ±5% for options
                new_price = last_price + random.uniform(-fluctuation, fluctuation)
                new_price = max(10.0, min(500.0, new_price))
            else:
                fluctuation = last_price * 0.001  # ±0.1% for NIFTY 50 or others
                new_price = last_price + random.uniform(-fluctuation, fluctuation)
                new_price = max(20000.0, min(30000.0, new_price)) if name == "NIFTY 50" else max(10.0, min(10000.0, new_price))

            MOCK_PRICE_CACHE[instrument_token] = new_price

        return {
            "instrument_token": instrument_token,
            "trading_symbol": trading_symbol,
            "exchange": exchange,
            "last_price": new_price,
            "mock": True
        }

    except Exception as e:
        logging.error(f"/get_price error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating mock price: {str(e)}")

@app.get("/nifty_option_price")
async def nifty_option_price():
    """
    Generate mock NIFTY 50 LTP and CE/PE option prices for the nearest weekly expiry.
    """
    try:
        if INSTRUMENT_DATA is None or INSTRUMENT_DATA.empty:
            success = await fetch_instruments()
            if not success or INSTRUMENT_DATA is None or INSTRUMENT_DATA.empty:
                await load_instrument_data()
            if INSTRUMENT_DATA is None or INSTRUMENT_DATA.empty:
                raise HTTPException(status_code=500, detail="Instrument data not loaded: No valid data source found.")

        # Validate required columns
        required_columns = {"segment", "name", "instrument_type", "strike", "expiry", "instrument_token", "tradingsymbol"}
        missing = required_columns - set(INSTRUMENT_DATA.columns)
        if missing:
            logging.error(f"Missing required columns in INSTRUMENT_DATA: {missing}")
            raise HTTPException(status_code=500, detail=f"Missing required columns: {missing}")

        # Generate mock NIFTY 50 LTP
        nifty_ltp = await get_nifty_spot_price()
        logging.info(f"Generated mock NIFTY 50 LTP: {nifty_ltp}")

        # Calculate target strike prices
        ce_strike = round((nifty_ltp + 50) / 50) * 50
        pe_strike = round((nifty_ltp - 50) / 50) * 50

        # Get nearest weekly expiry
        nearest_expiry = get_option_expiry("weekly")
        expiry_str = nearest_expiry.strftime("%Y-%m-%d")
        logging.info(f"Searching for CE strike {ce_strike} and PE strike {pe_strike} on expiry {expiry_str}")

        # Find CE and PE options
        try:
            ce_option = INSTRUMENT_DATA[
                (INSTRUMENT_DATA["segment"] == "NFO-OPT") &
                (INSTRUMENT_DATA["name"] == "NIFTY") &
                (INSTRUMENT_DATA["instrument_type"] == "CE") &
                (INSTRUMENT_DATA["strike"] == ce_strike) &
                (INSTRUMENT_DATA["expiry"] == expiry_str)
            ]
            pe_option = INSTRUMENT_DATA[
                (INSTRUMENT_DATA["segment"] == "NFO-OPT") &
                (INSTRUMENT_DATA["name"] == "NIFTY") &
                (INSTRUMENT_DATA["instrument_type"] == "PE") &
                (INSTRUMENT_DATA["strike"] == pe_strike) &
                (INSTRUMENT_DATA["expiry"] == expiry_str)
            ]
        except Exception as e:
            logging.error(f"Error filtering INSTRUMENT_DATA for options: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error filtering option data: {str(e)}")

        if ce_option.empty or pe_option.empty:
            logging.info(f"Fallback: Selecting closest strikes for CE {ce_strike} and PE {pe_strike}")
            try:
                ce_option = INSTRUMENT_DATA[
                    (INSTRUMENT_DATA["segment"] == "NFO-OPT") &
                    (INSTRUMENT_DATA["name"] == "NIFTY") &
                    (INSTRUMENT_DATA["instrument_type"] == "CE") &
                    (INSTRUMENT_DATA["expiry"] == expiry_str)
                ].sort_values(by="strike")
                ce_option = ce_option.iloc[(ce_option["strike"] - ce_strike).abs().argsort()[:1]] if not ce_option.empty else pd.DataFrame()

                pe_option = INSTRUMENT_DATA[
                    (INSTRUMENT_DATA["segment"] == "NFO-OPT") &
                    (INSTRUMENT_DATA["name"] == "NIFTY") &
                    (INSTRUMENT_DATA["instrument_type"] == "PE") &
                    (INSTRUMENT_DATA["expiry"] == expiry_str)
                ].sort_values(by="strike")
                pe_option = pe_option.iloc[(pe_option["strike"] - pe_strike).abs().argsort()[:1]] if not pe_option.empty else pd.DataFrame()
            except Exception as e:
                logging.error(f"Error in fallback option selection: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error in fallback option selection: {str(e)}")

            if ce_option.empty or pe_option.empty:
                logging.error(f"No NIFTY options found for strikes {ce_strike} (CE) and {pe_strike} (PE) on expiry {expiry_str}")
                raise HTTPException(status_code=404, detail=f"No matching NIFTY options found for strikes {ce_strike} (CE) and {pe_strike} (PE) on expiry {expiry_str}")

        # Validate symbols
        ce_symbol = ce_option["tradingsymbol"].iloc[0] if not ce_option.empty else None
        pe_symbol = pe_option["tradingsymbol"].iloc[0] if not pe_option.empty else None
        logging.info(f"Generated symbols: CE={ce_symbol}, PE={pe_symbol}")
        if not ce_symbol or not pe_symbol:
            logging.error("Invalid option symbols generated")
            raise HTTPException(status_code=400, detail="Invalid option symbols generated")

        # Generate mock LTPs for CE and PE options
        with MOCK_PRICE_CACHE_LOCK:
            ce_token = ce_option["instrument_token"].iloc[0]
            pe_token = pe_option["instrument_token"].iloc[0]
            ce_last_price = MOCK_PRICE_CACHE.get(ce_token, 200.0)
            pe_last_price = MOCK_PRICE_CACHE.get(pe_token, 200.0)

            # Apply fluctuations
            for token, last_price in [(ce_token, ce_last_price), (pe_token, pe_last_price)]:
                fluctuation = last_price * 0.05  # ±5%
                new_price = last_price + random.uniform(-fluctuation, fluctuation)
                new_price = max(10.0, min(500.0, new_price))
                MOCK_PRICE_CACHE[token] = new_price
            ce_ltp = MOCK_PRICE_CACHE[ce_token]
            pe_ltp = MOCK_PRICE_CACHE[pe_token]

        # Build response
        result = {
            "nifty_ltp": float(nifty_ltp),
            "options": [
                {
                    "instrument_token": int(ce_option["instrument_token"].iloc[0]),
                    "trading_symbol": str(ce_symbol),
                    "strike": float(ce_option["strike"].iloc[0]),
                    "option_type": "CE",
                    "expiry": str(ce_option["expiry"].iloc[0]),
                    "last_price": float(ce_ltp)
                },
                {
                    "instrument_token": int(pe_option["instrument_token"].iloc[0]),
                    "trading_symbol": str(pe_symbol),
                    "strike": float(pe_option["strike"].iloc[0]),
                    "option_type": "PE",
                    "expiry": str(pe_option["expiry"].iloc[0]),
                    "last_price": float(pe_ltp)
                }
            ],
            "mock": True
        }

        logging.info(f"Returning mock response: {result}")
        return result

    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions
    except Exception as e:
        logging.error(f"/nifty_option_price error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating mock option prices: {str(e)}")

@app.get("/logs", dependencies=[Depends(verify_api_key)])
async def get_error_logs():
    """
    Return the contents of the mock error log file.
    """
    log_file = "mock_error.log"
    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail="Log file not found.")
    async with aiofiles.open(log_file, "r") as f:
        log_content = await f.read()
    return {"log": log_content}

@app.get("/india_vix")
async def get_india_vix():
    """
    Generate mock India VIX value.
    """
    try:
        logging.info("Generating mock India VIX value.")
        base_vix = 15.0  # Typical India VIX value
        fluctuation = base_vix * 0.05  # ±5% or ±0.75 points
        vix_value = base_vix + random.uniform(-fluctuation, fluctuation)
        vix_value = max(10.0, min(30.0, vix_value))  # Constrain to 10–30
        return {
            "symbol": "INDIA VIX",
            "india_vix": vix_value,
            "mock": True
        }

    except Exception as e:
        logging.error(f"/india_vix error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating mock India VIX: {str(e)}")