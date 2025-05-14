# FINAL Phase 1 Execution Plan: GraphRAG Tutor
# (Q&A Focus, Annotation, Save Type Choice - Full Supabase Integration)

**Document Version:** 1.0 (FINAL - Execute based on this plan)
**Date:** 2025-04-18
**Status:** Analysis Complete. Ready for Implementation Planning & Execution.

**Document Purpose:** Provides the definitive, highly detailed, micro-process level execution plan for Phase 1. Focuses on a polished RAG Q&A experience with **persistent storage and authentication via Supabase**. Users log in, ask questions, annotate responses, and save items (notes/flashcards) to their account. Includes KG placeholder. Requires parallel frontend/backend development.

**Overall Phase 1 Objective:** Implement a polished **RAG Q&A panel** integrated with a **Supabase backend** for **user authentication** and **persistent storage** of annotated notes/flashcards. Implement a **placeholder panel for KG visualization**. Arrange in a basic layout. Define and implement necessary backend APIs and integrate frontend fully.

**Key Features & Scope (Phase 1):**
* Polished Q&A interaction panel with smooth loading/error state transitions.
* User annotations on AI responses.
* Choice to save annotated Q&A as 'note' or 'flashcard'.
* **Storage:** Uses **Supabase (Postgres)** via backend API for saved items.
* **Authentication:** Basic email/password user login/signup via Supabase Auth integrated. Secure token handling.
* Placeholder for Knowledge Graph visualization (data fetch only).
* Basic fixed layout and header with conditional auth elements.
* **Backend:** Supabase project setup, secure key management, Postgres schema definition (incl. RLS), FastAPI endpoints (auth, item CRUD) with validation and error handling.
* **Frontend:** React UI components (Shadcn UI), Zustand state management (incl. auth state), API service calls with token handling, form handling.
* **Testing:** Unit, integration, and E2E test coverage for both frontend and backend.
* **Deferred:** Document listing/viewing, citation click-through, advanced auth (social logins, MFA, password reset), dedicated flashcard review UI/logic, KG visualization rendering, admin interfaces, advanced error recovery, comprehensive performance optimization.

---

## Prerequisites & Assumptions

* **Phase 0 Complete:** React/Vite/Tailwind/Shadcn/Zustand setup done. Base FastAPI backend structure exists. Backend API `/generate`, `/graph` stable and performant enough for initial use.
* **Admin-Seeded Knowledge:** KG managed by admin.
* **Supabase Project Ready:** Supabase project created. Consider separate projects for Dev/Staging/Prod environments later. URL and API keys secured.
* **Backend Stack:** Python 3.9+, FastAPI, Pydantic, `supabase-py`, `python-dotenv`, `uvicorn`.
* **Frontend Stack:** React 18+, Vite, Zustand, `fetch`/`axios`, Shadcn/ui, Tailwind CSS.
* **Tooling:** Node.js/npm for frontend, Python environment management (e.g., venv, Poetry), Git for version control.
* **Environment Variables:** Secure handling of Supabase keys, API URLs, potential JWT secrets via `.env` files is critical.

## General Implementation Guidance

* **STOP ANALYSIS, START BUILDING:** This plan is the blueprint. Focus now shifts to implementation task breakdown and execution.
* **Parallel Development:** Requires tight coordination. Define API contract **FIRST** using OpenAPI/Swagger spec generated from FastAPI/Pydantic models. Backend endpoints needed before full frontend integration.
* **Task Breakdown:** Break down these micro-processes into smaller sprint tasks/user stories.
* **AI Collaboration:** Use specific prompts derived from micro-processes. **Mandatory: Critically review, rigorously test, and refactor ALL AI-generated code.** Pay close attention to security, error handling, edge cases, and state logic correctness.
* **Environment Variables:** Use `.env` files (add to `.gitignore`). Backend: Load via `python-dotenv` and Pydantic `BaseSettings`. Frontend: Use Vite's `import.meta.env` (prefix variables with `VITE_`). Ensure NO keys are committed to Git.
* **Polished UI:** Use Shadcn UI thoughtfully. Implement smooth transitions (e.g., `framer-motion` for subtle animations on message load/save confirmation). Use skeleton loaders (`Skeleton`) during content fetching. Provide clear visual feedback (spinners in buttons, toast notifications for errors/success). Use appropriate icons (Lucide icons bundled with Shadcn).
* **Testing Strategy:** Implement unit, integration, and E2E tests as specified. Aim for reasonable coverage focusing on critical paths (auth, core Q&A loop, saving).
* **Security:** Prioritize security: HTTPS, input validation (Pydantic), parameterized SQL (handled by Supabase client), correct RLS policies, secure JWT handling (storage & transmission), dependency vulnerability scanning.

---

## Phase 1 Tasks & Micro-Processes

### **Part A: Backend Setup & API Development (Python/FastAPI + Supabase)**

* **Task A.1: Supabase Project Setup & Environment**
    * **Goal:** Configure Supabase project, secure keys, set up env vars.
    * **Guidance:**
        * Create Supabase project(s). Securely store Project URL, `anon` key, `service_role` key.
        * **Backend `.env`:**
          ```# .env (backend - add to .gitignore!)
          SUPABASE_URL="YOUR_PROJECT_URL"
          SUPABASE_SERVICE_KEY="YOUR_SERVICE_ROLE_KEY"
          # Optional: If using custom JWT signing beyond Supabase defaults
          # JWT_SECRET="YOUR_STRONG_SECRET_KEY"
          # ALGORITHM="HS256"
          # ACCESS_TOKEN_EXPIRE_MINUTES=30
          ```
        * **Backend Config (`config.py`):** Use Pydantic `BaseSettings` to load `.env`.
          ```python
          from pydantic_settings import BaseSettings
          class Settings(BaseSettings):
              supabase_url: str
              supabase_service_key: str
              # jwt_secret: str = "default_secret_change_me" # Example
              class Config:
                  env_file = '.env'
          settings = Settings()
          ```
        * Ensure backend initializes Supabase client using these settings.

* **Task A.2: Define Database Schema (Supabase Postgres)**
    * **Goal:** Create `saved_items` table with specific types and RLS.
    * **Guidance:**
        * Use Supabase SQL Editor or DB migrations (like Alembic if integrating with FastAPI).
        * **SQL Schema:**
          ```sql
          -- Ensure required extensions like pgcrypto for gen_random_uuid() are enabled
          CREATE EXTENSION IF NOT EXISTS "pgcrypto";

          CREATE TABLE public.saved_items (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE, -- Link to Supabase Auth users
              created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
              question TEXT NOT NULL,
              answer TEXT NOT NULL,
              citations JSONB, -- Store as array of objects: [{"id": "...", "text": "..."}]
              annotation TEXT, -- Allowed to be NULL
              type TEXT NOT NULL CHECK (type IN ('note', 'flashcard')) -- Enforce specific types
              -- Add other metadata if needed (e.g., session_id, model_used)
          );

          -- Indexes for performance
          CREATE INDEX idx_saved_items_user_id_created_at ON public.saved_items(user_id, created_at DESC); -- Common query pattern
          CREATE INDEX idx_saved_items_type ON public.saved_items(type); -- If filtering by type becomes common

          -- Enable Row Level Security
          ALTER TABLE public.saved_items ENABLE ROW LEVEL SECURITY;
          ALTER TABLE public.saved_items FORCE ROW LEVEL SECURITY; -- Important for security

          -- RLS Policies (Ensure these match user roles/needs)
          CREATE POLICY "Allow ALL access for service_role" ON public.saved_items FOR ALL USING (auth.role() = 'service_role'); -- Allow backend full access
          CREATE POLICY "Allow individual read access" ON public.saved_items FOR SELECT USING (auth.uid() = user_id);
          CREATE POLICY "Allow individual insert access" ON public.saved_items FOR INSERT WITH CHECK (auth.uid() = user_id);
          CREATE POLICY "Allow individual update access" ON public.saved_items FOR UPDATE USING (auth.uid() = user_id); -- Optional for Phase 1
          CREATE POLICY "Allow individual delete access" ON public.saved_items FOR DELETE USING (auth.uid() = user_id); -- Optional for Phase 1
          ```
        * **Verification:** Use Supabase dashboard to confirm table structure, types, indexes, and RLS policies are active and correct.

* **Task A.3: Backend Authentication Integration**
    * **Goal:** Implement secure signup/login using Supabase Auth.
    * **File(s):** `api_server.py`, `auth/router.py`, `auth/schemas.py`, `auth/dependencies.py`.
    * **Guidance:**
        * Initialize Supabase client (use `service_role` key for backend actions like signup if needed, or manage users appropriately).
        * **Pydantic Schemas (`auth/schemas.py`):** Define `UserCreate`, `UserLogin`, `TokenResponse`, `UserResponse` as before. Add validation (e.g., password length).
        * **Authentication Dependency (`auth/dependencies.py`):**
          ```python
          from fastapi import Depends, HTTPException, status
          from fastapi.security import OAuth2PasswordBearer
          from supabase_py_async import AsyncClient # Or sync version
          from jose import JWTError, jwt
          # ... import settings, schemas, get_supabase_client ...

          # If using Supabase session tokens directly:
          async def get_current_user_supabase(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/auth/login")), supabase: AsyncClient = Depends(get_supabase_client)):
              try:
                  # Use Supabase client to validate the token and get user
                  res = await supabase.auth.get_user(token)
                  if not res.user:
                      raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials", headers={"WWW-Authenticate": "Bearer"})
                  # Map Supabase user to your UserResponse schema or return relevant ID
                  return UserResponse(id=res.user.id, email=res.user.email) # Example mapping
              except Exception as e: # Catch specific Supabase auth errors
                  raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Could not validate credentials: {e}", headers={"WWW-Authenticate": "Bearer"})

          # If using custom JWTs signed by backend (more complex, likely not needed if using Supabase tokens):
          # Define get_current_user_custom_jwt using jose library, settings.JWT_SECRET etc.

          # Choose ONE method (Supabase token validation recommended for simplicity)
          get_current_user = get_current_user_supabase
          ```
        * **FastAPI Router (`auth/router.py`):** Implement `/signup`, `/login` endpoints using defined schemas. Handle specific Supabase errors (e.g., `AuthApiError` for existing user, invalid credentials) and return appropriate HTTP status codes (400, 401, 409). Ensure passwords are not logged. Return `TokenResponse` on successful login. Add `/logout` if needed (calls `supabase.auth.sign_out`). Add protected `/me` endpoint using `Depends(get_current_user)`.

* **Task A.4: Backend CRUD API for Saved Items**
    * **Goal:** Implement secure endpoints to create/read saved items.
    * **File(s):** `api_server.py`, `items/router.py`, `items/schemas.py`.
    * **Guidance:**
        * **Pydantic Schemas (`items/schemas.py`):** Define `Citation`, `ItemBase`, `ItemCreate`, `ItemResponse` as before. Add validation (e.g., ensure `type` is 'note' or 'flashcard').
        * **FastAPI Router (`items/router.py`):**
          * Implement `POST /` endpoint: Use `Depends(get_current_user)` for protection. Extract `user_id`. Validate incoming `ItemCreate` data using Pydantic. Insert into `saved_items` table via Supabase client. Handle potential database errors (e.g., constraint violations, connection errors) gracefully. Return `ItemResponse` of the created item.
          * Implement `GET /` endpoint: Use `Depends(get_current_user)`. Extract `user_id`. Fetch items for that user from Supabase, ordering by `created_at DESC`. Return `List[ItemResponse]`. Handle potential errors.

* **Testing A.1 - A.4 (Backend - Highly Detailed):**
    * **Unit Tests (Pytest):**
        * Mock `supabase-py` client thoroughly using `unittest.mock`.
        * Test `signup` endpoint logic for success, user exists error, other Supabase errors.
        * Test `login` endpoint logic for success, invalid credentials error.
        * Test `create_item` endpoint: Correct data insertion format, handling of optional fields (annotation, citations), auth protection.
        * Test `read_items` endpoint: Correct data retrieval, ordering, auth protection, empty list scenario.
        * Test Pydantic schema validation (valid and invalid inputs).
        * Test `get_current_user` dependency (mocking token validation).
    * **Integration Tests (Pytest + TestClient):**
        * Use separate test Supabase project/schema if possible, manage via fixtures (`pytest-asyncio` if using async).
        * Test full Auth flow: Signup -> Login -> Access protected endpoint (`/items` GET) -> Logout.
        * Test Item CRUD: Login -> Create Item -> Get Items (verify created item exists) -> (If implemented) Update Item -> Get Items -> Delete Item -> Get Items (verify deleted).
        * Test RLS: Signup/Login User A -> Create Item A -> Signup/Login User B -> Attempt to GET/Update/Delete Item A as User B (expect 401/403/404 based on policy).
    * **Security Checks:** Review logs (ensure no sensitive data leaked). Manually test endpoints with invalid/missing tokens. Check RLS policies in Supabase UI.

### **Part B: Frontend Development (React + Zustand + Supabase Integration)**

* **Task B.1: Implement Core Panel Components (Static Structure)**
    * **Goal:** Define static structure with testing IDs.
    * **File:** `src/components/panels/RAGQAPanel.jsx`, `src/components/panels/KGVisualizationPanel.jsx`
    * **Guidance:** Structure components. Add `data-testid` attributes to key elements (message list, query input, send button, annotation inputs, save buttons) for testing.

* **Task B.2: Implement Basic Fixed Layout & Navigation**
    * **Goal:** Arrange panels, add header with auth status.
    * **File:** `src/layouts/MainLayout.jsx`, `src/App.jsx`, new `src/components/layout/Header.jsx`.
    * **Guidance:** Use grid layout. Create `Header` component. Render `MainLayout` within `App` (potentially via router). `Header` shows app title and conditionally displays Username/Logout button (using state from `useAuthStore`) or Login/Signup links.

* **Task B.3: Frontend Authentication UI & State**
    * **Goal:** Implement login/signup forms, state management, routing.
    * **File(s):** `src/components/auth/LoginForm.jsx`, `src/components/auth/SignupForm.jsx`, `src/store/useAuthStore.js`, `App.jsx` (or `src/router/index.jsx`).
    * **Guidance:**
        * **Forms:** Use Shadcn `Card`, `Input`, `Button`, `Label`. Implement controlled inputs (`useState`). Add form validation (e.g., basic required fields, email format). Display loading state on buttons during submission. Display error messages (`Alert`) within forms on failure. Call `signIn`/`signUp` actions from `useAuthStore` `onSubmit`.
        * **`useAuthStore`:** State `user`, `sessionToken`, `authLoading` (boolean), `authError` (string | null). Actions `signIn`, `signUp`, `signOut`, `setSession`, `checkInitialSession`. Actions should set `authLoading`, call `api.js` functions, handle success/error by updating state (`user`, `sessionToken`, `authError`), clear error on new attempt, unset `authLoading`.
        * **Routing:** Use `react-router-dom`. Create protected routes for `MainLayout` that check `sessionToken` in `useAuthStore`. Redirect to `/login` if no token. Have `/login` and `/signup` routes rendering the forms. `App.jsx` sets up router.
        * **Session Persistence:** `checkInitialSession` action runs on app mount (`useEffect` in `App.jsx`). Tries to load token from `localStorage`. If found, potentially calls backend `/me` to validate and get user info, then updates store state via `setSession`.

* **Task B.4: Define Frontend API Service Functions (Calls Backend)**
    * **Goal:** Implement functions to call backend, handle auth headers, manage errors.
    * **File:** `src/services/api.js`
    * **Guidance:**
        * Use `axios` for easier request/response interception and error handling, or stick with `fetch`.
        * Implement `setAuthHeader(token)` / `clearAuthHeader` helpers for `axios` instance or `fetch` options.
        * `login`, `signup`, `logout`, `getCurrentUser`: Call backend `/auth/...`. `login` stores token (e.g., `localStorage.setItem('authToken', token)`), calls `setAuthHeader`. `logout` clears token, calls `clearAuthHeader`.
        * `saveNewItem(itemData, token)`: Calls `POST /items`. Expects `itemData` matching backend `ItemCreate` schema.
        * `fetchItems(token)`: Calls `GET /items`.
        * `generateQuery`, `getGraphData`: Add token via auth header helper.
        * **Error Handling:** Wrap calls in `try...catch`. Catch specific HTTP errors (401, 403, 404, 500). Extract error messages from backend response body if available, otherwise return generic error.

* **Task B.5: Define Zustand Stores (Updated for Backend)**
    * **Goal:** Finalize stores for backend interaction.
    * **File(s):** `src/store/useChatStore.js`, `src/store/useGraphStore.js`, `useAuthStore.js`.
    * **Guidance:**
        * `useChatStore`: Refine state (`messages`, `qaLoading`, `qaError`, `savedItems`, `saveLoading`, `saveError`, `fetchSavedLoading`, `fetchSavedError`). Actions (`sendMessage`, `saveItem`, `fetchSavedItems`) now fully interact with `api.js`, setting relevant loading/error states within `try...catch...finally` blocks. Ensure `fetchSavedItems` result populates `savedItems`.
        * `useGraphStore`: Similar pattern for `fetchGraphData` with loading/error state.
        * `useAuthStore`: Actions interact with `api.js` auth functions, manage token in `localStorage`, update `user`/`sessionToken` state.

* **Task B.6: Connect `RAGQAPanel` (Full Integration)**
    * **Goal:** Fully connect Q&A panel, handle state/API calls, determine saved status.
    * **File:** `src/components/panels/RAGQAPanel.jsx`
    * **Guidance:**
        * Connect to `useChatStore` and `useAuthStore`.
        * Manage `inputValue` and `annotations` state.
        * Call `sendMessage` action on submit.
        * **Determining `isSaved`:** After `fetchSavedItems` populates `useChatStore.savedItems`, implement logic within the component (perhaps using `useMemo` or `useEffect` watching `messages` and `savedItems`) to efficiently check if a displayed message corresponds to an item in `savedItems`. A robust way might involve adding a unique identifier (e.g., a temporary frontend ID or one generated by the backend on Q&A response) to messages and trying to match based on that, or falling back to content matching if necessary. Update the message object locally (`messages` might need modification or a parallel mapping) to reflect `isSaved`.
        * Render UI based on store states: `qaLoading` (spinner near input), `saveLoading` (spinner on save buttons), `qaError` / `saveError` (`Alert` components). Disable save buttons based on `isSaved` status or `saveLoading`.
        * Conditionally render based on `useAuthStore.sessionToken` if needed (though main layout protection is primary).
        * Trigger `fetchSavedItems` via an effect when auth state changes to logged-in.

* **Task B.7: Connect `KGVisualizationPanel` (Full Integration)**
    * **Goal:** Connect KG panel to fetch data via backend.
    * **File:** `src/components/panels/KGVisualizationPanel.jsx`
    * **Guidance:** Use `useGraphStore`. Call `fetchGraphData`. Display loading/error/status from store.

* **Testing B.1 - B.7 (Frontend - Highly Detailed):**
    * **Unit Tests (Vitest/RTL):**
        * Test utility functions (e.g., token handlers).
        * Test components in isolation: rendering based on props, handling user input (`fireEvent`), calling mock functions passed as props. Test Login/Signup forms validation. Test message rendering. Test display of loading/error states based on props.
        * Test Zustand stores: Mock `api.js`. Verify actions update state correctly (loading, error, data). Verify selectors return correct data. Test `checkInitialSession` logic with mock `localStorage`.
    * **Integration Tests (Vitest/RTL, potentially MSW for API mocking):**
        * Test components interacting with stores. E.g., test `RAGQAPanel` dispatching actions and updating based on store changes. Mock the API layer (`api.js`) using tools like Mock Service Worker (MSW) to simulate backend responses (success, errors). Test the full Q&A -> Annotate -> Save flow against the mocked API. Test the Login/Signup flow against mocked API.
    * **E2E Tests (Playwright/Cypress):**
        * **Setup:** Requires running frontend dev server AND backend dev server (connected to test Supabase instance if possible, or fully mocked backend).
        * **Auth:** `signup` -> `logout` -> `login` -> verify access to main layout -> refresh & verify still logged in -> `logout`.
        * **Core Loop:** Login -> Ask Question -> Verify Response -> Add Annotation -> Click "Save Note" -> Verify UI feedback -> (If possible via test hooks or separate check) Verify data in Supabase DB -> Refresh -> Verify item conceptually marked as saved. Repeat for "Save Flashcard".
        * **Error Handling:** Test scenarios where backend API returns errors (4xx, 5xx) for Q&A, saving, fetching. Test network down scenarios. Verify user-friendly error messages.
        * **Cross-Component:** Test interaction if applicable (though less likely in this focused Phase 1).

---

## Final Phase 1 Summary (Supabase Integration - Expanded Scope)

This **final, detailed execution plan** outlines the development of the core Q&A functionality, fully integrated with Supabase for persistence and authentication, including annotation and save type choices. It requires significant, coordinated **backend and frontend development effort within Phase 1**. While delivering a robust foundation, this represents a considerable undertaking compared to the initial frontend-only scope. **No further analysis of this Phase 1 plan is required.** Proceed with task breakdown, sprint planning, and implementation based on this document. Manage dependencies and testing rigorously.