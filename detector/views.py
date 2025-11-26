from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpRequest
from django.conf import settings
from pathlib import Path
import json
from .services.predict import load_predictor, predict_text
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User


_PREDICTOR = None


def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            _PREDICTOR = load_predictor()
        except Exception as e:
            # Log error but return empty predictor
            import traceback
            print(f"Error loading predictor: {e}")
            traceback.print_exc()
            _PREDICTOR = load_predictor()  # Will create empty predictor
    return _PREDICTOR


def _reload_predictor():
    """Force reload predictor - useful after retraining"""
    global _PREDICTOR
    _PREDICTOR = None
    return _get_predictor()


from django.core.paginator import Paginator
from .models import Search

@login_required
@ensure_csrf_cookie
def home(request: HttpRequest):
    all_searches = Search.objects.filter(user=request.user).order_by('-created_at')
    paginator = Paginator(all_searches, 10)  # Show 10 searches per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'detector/home.html', {'page_obj': page_obj})


@login_required
@csrf_exempt
def api_predict(request: HttpRequest):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        data = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    text = data.get('text', '')
    model = data.get('model', 'best')
    if not text:
        return JsonResponse({'error': 'text is required'}, status=400)

    try:
        predictor = _get_predictor()
        result = predict_text(predictor, text, model)
        
        # Ensure result has required fields
        if 'label' not in result or result['label'] in ['unknown', 'error']:
            # Try to reload predictor in case models were just trained
            predictor = _reload_predictor()
            result = predict_text(predictor, text, model)
        
        # Save search to CSV
        searches_dir: Path = Path(settings.SEARCHES_DIR)
        searches_dir.mkdir(parents=True, exist_ok=True)
        out_path = searches_dir / 'searches.csv'
        header_needed = not out_path.exists()
        with out_path.open('a', encoding='utf-8') as f:
            if header_needed:
                f.write('text,model,label,score\n')
            # naive escaping for commas/newlines
            safe_text = text.replace('\n', ' ').replace('\r', ' ').replace(',', ' ')
            label = result.get('label', 'unknown')
            score = result.get('score', 0.0)
            actual_model = result.get('model', model)
            f.write(f"{safe_text},{actual_model},{label},{score}\n")

        # Save search to database for per-user history
        try:
            Search.objects.create(
                user=request.user,
                text=text,
                model=model,
                label=result.get('label', 'unknown'),
                score=float(result.get('score', 0.0)),
            )
        except Exception:
            # If DB save fails, continue returning prediction result
            pass

        return JsonResponse(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f'Prediction error: {str(e)}', 'label': 'error', 'score': 0.0}, status=500)

def auth_page(request: HttpRequest):
    """Combined login/signup page"""
    if request.user.is_authenticated:
        return redirect('home')
    return render(request, 'detector/auth.html')


def signup_view(request: HttpRequest):
    """Handle user signup"""
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '').strip()
        password_confirm = request.POST.get('password_confirm', '').strip()
        
        # Validation
        if not username or not email or not password:
            return JsonResponse({'error': 'All fields are required'}, status=400)
        
        if password != password_confirm:
            return JsonResponse({'error': 'Passwords do not match'}, status=400)
        
        if len(password) < 8:
            return JsonResponse({'error': 'Password must be at least 8 characters'}, status=400)
        
        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': 'Username already exists'}, status=400)
        
        if User.objects.filter(email=email).exists():
            return JsonResponse({'error': 'Email already exists'}, status=400)
        
        # Create user
        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )
            login(request, user)
            return JsonResponse({'success': True, 'message': 'Account created successfully!'})
        except Exception as e:
            return JsonResponse({'error': f'Error creating account: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def login_view(request: HttpRequest):
    """Handle user login"""
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        
        if not username or not password:
            return JsonResponse({'error': 'Username and password are required'}, status=400)
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({'success': True, 'message': 'Login successful!'})
        else:
            return JsonResponse({'error': 'Invalid username or password'}, status=401)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def logout_view(request: HttpRequest):
    """Handle user logout"""
    from django.contrib.auth import logout
    logout(request)
    return redirect('auth_page')
