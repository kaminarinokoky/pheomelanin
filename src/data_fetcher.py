import requests

def get_order_book(src_currency, dst_currency):
    """
    Fetches the order book for a given market (e.g., BTC_IRR).
    
    :param src_currency: The source currency (e.g., "btc").
    :param dst_currency: The destination currency (e.g., "rls" for IRR).
    :return: The order book data as a dictionary.
    """
    url = "https://api.nobitex.ir/market/orders/list"
    data = {
        "order": "price",
        "srcCurrency": src_currency,
        "dstCurrency": dst_currency
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get order book: {response.status_code}")

def get_recent_trades(src_currency, dst_currency):
    """
    Fetches recent trades for a given market (e.g., BTC_IRR).
    
    :param src_currency: The source currency (e.g., "btc").
    :param dst_currency: The destination currency (e.g., "rls" for IRR).
    :return: The recent trades data as a dictionary.
    """
    url = "https://api.nobitex.ir/market/trades/list"
    data = {
        "srcCurrency": src_currency,
        "dstCurrency": dst_currency
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get recent trades: {response.status_code}")

# Example usage
if __name__ == "__main__":
    src_currency = "btc"
    dst_currency = "rls"  # "rls" is the code for Iranian Rial (IRR) on Nobitex
    
    # Fetch and print the order book
    try:
        order_book = get_order_book(src_currency, dst_currency)
        print("Order Book:")
        print(order_book)
    except Exception as e:
        print(e)
    
    # Fetch and print recent trades
    try:
        recent_trades = get_recent_trades(src_currency, dst_currency)
        print("Recent Trades:")
        print(recent_trades)
    except Exception as e:
        print(e)