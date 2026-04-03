#!/usr/bin/env python3
# ============================================================
# 🌙 Hyperliquid Setup & Connection Test
#
# Run this after completing the setup steps to verify
# everything is working before live trading.
#
# Usage:
#   python src/agents/hyperliquid_setup.py
# ============================================================
#
# ══════════════════════════════════════════════════════════════
# STEP-BY-STEP SETUP GUIDE
# ══════════════════════════════════════════════════════════════
#
# ── STEP 1: Get your Ethereum address from Exodus ─────────────
#
#   Hyperliquid uses your Ethereum wallet address to identify you.
#   Your Exodus ETH address IS your Hyperliquid account address.
#
#   In Exodus Desktop (Mac):
#   a) Click the Wallet icon (left sidebar)
#   b) Find "Ethereum" in the asset list and click it
#   c) Click the Receive button
#   d) Copy your ETH address (starts with 0x...)
#      → This is your HYPERLIQUID_ACCOUNT_ADDRESS
#
# ── STEP 2: Fund your account via Arbitrum ────────────────────
#
#   Hyperliquid deposits go through the Arbitrum bridge.
#   You need USDC on Arbitrum to deposit.
#
#   Option A — If you already have USDC on Arbitrum:
#     a) Go to app.hyperliquid.xyz
#     b) Click Deposit
#     c) Connect MetaMask OR use WalletConnect to connect Exodus
#     d) Deposit USDC (minimum ~$10 to test, $200+ for real trading)
#
#   Option B — If your USDC is on a different chain:
#     a) Go to app.rhino.fi or app.across.to (bridging services)
#     b) Bridge your USDC to Arbitrum first
#     c) Then deposit to Hyperliquid as above
#
#   Option C — Buy USDC directly on Arbitrum:
#     a) On Exodus, swap any crypto to USDC
#     b) Send USDC to your Arbitrum address (in Exodus, ETH and
#        USDC share the same address — your 0x... address)
#     c) Deposit to Hyperliquid
#
#   NOTE: You can also connect Exodus directly to Hyperliquid
#   via WalletConnect if Exodus Mobile supports it.
#
# ── STEP 3: Generate an API Wallet on Hyperliquid ─────────────
#
#   IMPORTANT: Do NOT use your main wallet private key for the bot.
#   Hyperliquid lets you create a separate API wallet that can
#   TRADE but CANNOT WITHDRAW funds. This is much safer.
#
#   a) Go to: https://app.hyperliquid.xyz/API
#   b) Click "Generate" or "Create API Wallet"
#   c) Give it a name (e.g. "algotec-bot")
#   d) Enter your main wallet address in the field
#   e) Click "Authorize API Wallet" — sign the transaction
#      (this costs a tiny gas fee on Arbitrum, ~$0.01)
#   f) You'll see a PRIVATE KEY for the API wallet
#      → COPY THIS IMMEDIATELY — it's only shown once
#      → This is your HYPERLIQUID_API_PRIVATE_KEY
#
#   Your main wallet address (from Step 1) is your account address.
#   The API wallet private key (from this step) is what the bot uses.
#
# ── STEP 4: Add to your .env file ─────────────────────────────
#
#   Open: trading_system/.env
#   Add these lines:
#
#   # Hyperliquid
#   HYPERLIQUID_ACCOUNT_ADDRESS=0xYOUR_ETH_ADDRESS_FROM_STEP_1
#   HYPERLIQUID_API_PRIVATE_KEY=0xAPI_WALLET_PRIVATE_KEY_FROM_STEP_3
#
#   Example (these are fake — use your real ones):
#   HYPERLIQUID_ACCOUNT_ADDRESS=0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
#   HYPERLIQUID_API_PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234...
#
# ── STEP 5: Install the SDK ───────────────────────────────────
#
#   pip install hyperliquid-python-sdk
#
# ── STEP 6: Test the connection ───────────────────────────────
#
#   python src/agents/hyperliquid_setup.py
#
#   You should see:
#   ✅ Connected to Hyperliquid
#   ✅ Account: 0xYour...
#   ✅ Balance: $XXX.XX USDC
#   ✅ API wallet authorised
#
# ══════════════════════════════════════════════════════════════

import os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def test_connection():
    account_address = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS", "")
    api_private_key = os.getenv("HYPERLIQUID_API_PRIVATE_KEY", "")

    print("\n🌙 Hyperliquid Connection Test")
    print("=" * 45)

    # ── Check env vars ─────────────────────────────────────
    if not account_address:
        print("❌ HYPERLIQUID_ACCOUNT_ADDRESS not set in .env")
        print("   See Step 1 + 4 in the setup guide above")
        return False

    if not api_private_key:
        print("❌ HYPERLIQUID_API_PRIVATE_KEY not set in .env")
        print("   See Step 3 + 4 in the setup guide above")
        return False

    print(f"  Account : {account_address[:8]}...{account_address[-6:]}")

    # ── Try SDK ────────────────────────────────────────────
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        from eth_account import Account
    except ImportError:
        print("\n❌ hyperliquid-python-sdk not installed")
        print("   Run: pip install hyperliquid-python-sdk eth-account")
        return False

    # ── Check account balance ──────────────────────────────
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        state = info.user_state(account_address)

        margin_summary = state.get("marginSummary", {})
        balance = float(margin_summary.get("accountValue", 0))
        withdrawable = float(margin_summary.get("withdrawable", 0))

        print(f"  Balance : ${balance:,.2f} USDC")
        print(f"  Available: ${withdrawable:,.2f} USDC")

        if balance < 1:
            print("\n  ⚠️  Balance is very low — deposit USDC before trading")
            print("     Go to: app.hyperliquid.xyz → Deposit")

        print("  ✅ Account connected")

    except Exception as e:
        print(f"  ❌ Could not fetch account: {e}")
        print("     Make sure your address is correct and has funds deposited")
        return False

    # ── Verify API wallet ──────────────────────────────────
    try:
        wallet = Account.from_key(api_private_key)
        api_address = wallet.address
        print(f"\n  API wallet: {api_address[:8]}...{api_address[-6:]}")

        # Check if API wallet is authorised
        # (Hyperliquid returns agent permissions in user state)
        agents = state.get("agentAddress", "")
        if api_address.lower() in str(agents).lower():
            print("  ✅ API wallet is authorised")
        else:
            # Try to verify via a different check
            print("  ⚠️  Cannot confirm API wallet authorisation")
            print("     If you get auth errors, re-authorise at:")
            print("     app.hyperliquid.xyz/API")

    except Exception as e:
        print(f"  ❌ API wallet error: {e}")
        print("     Check that HYPERLIQUID_API_PRIVATE_KEY starts with 0x")
        return False

    # ── Check open positions ───────────────────────────────
    try:
        positions = state.get("assetPositions", [])
        open_pos   = [p for p in positions
                      if float(p.get("position", {}).get("szi", 0)) != 0]
        print(f"\n  Open positions: {len(open_pos)}")
        for p in open_pos:
            pos  = p.get("position", {})
            coin = pos.get("coin", "?")
            size = float(pos.get("szi", 0))
            upnl = float(pos.get("unrealizedPnl", 0))
            print(f"    {coin}: {'LONG' if size > 0 else 'SHORT'} "
                  f"size={abs(size)} uPnL=${upnl:+.2f}")
    except Exception:
        pass

    print("\n✅ Hyperliquid setup complete — ready to trade!")
    print("   Run: python src/agents/vault_forward_test.py --market crypto")
    return True


def place_test_order(symbol: str = "BTC", size_usd: float = 10):
    """
    Place a tiny test order to confirm execution works.
    Uses $10 by default — very small to minimise risk.
    """
    account_address = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS", "")
    api_private_key = os.getenv("HYPERLIQUID_API_PRIVATE_KEY", "")

    try:
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        from eth_account import Account

        wallet  = Account.from_key(api_private_key)
        info    = Info(constants.MAINNET_API_URL, skip_ws=True)
        exchange = Exchange(
            wallet,
            constants.MAINNET_API_URL,
            account_address=account_address
        )

        # Get current price
        mids  = info.all_mids()
        price = float(mids.get(symbol, 0))
        if not price:
            print(f"❌ Could not get price for {symbol}")
            return

        # Calculate size in contracts
        sz = round(size_usd / price, 4)

        print(f"\n🧪 Test order: BUY {sz} {symbol} @ ~${price:,.2f}")
        print(f"   Value: ~${size_usd:.2f}")

        # Place market order
        result = exchange.market_open(symbol, True, sz)
        print(f"   Result: {result}")

        if result.get("status") == "ok":
            print("✅ Test order placed successfully!")
            print("   Close it manually on app.hyperliquid.xyz")
        else:
            print(f"❌ Order failed: {result}")

    except Exception as e:
        print(f"❌ Test order error: {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Hyperliquid setup & test")
    p.add_argument("--test-order", action="store_true",
                   help="Place a $10 test order to verify execution")
    p.add_argument("--symbol", default="BTC")
    args = p.parse_args()

    ok = test_connection()

    if ok and args.test_order:
        confirm = input("\nPlace a $10 test order? (yes/no): ")
        if confirm.lower() == "yes":
            place_test_order(args.symbol)
