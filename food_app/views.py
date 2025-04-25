from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import FoodItem, RecommendationHistory
from .utils import get_recommendations
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json

#login view
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('home')
        messages.error(request, 'Invalid credentials')
    return render(request, 'login.html')

#signup view
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            if user:
                login(request, user)
                return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

#logout view
def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def home_view(request):
    food_groups = FoodItem.objects.values('food_group').distinct()
    food_items = FoodItem.objects.all()
    selected_group = request.GET.get('food_group', '')

    if selected_group:
        food_items = food_items.filter(food_group=selected_group)

    if request.method == 'POST':
        food_name = request.POST.get('food_item')
        if not food_name:
            messages.error(request, 'Please select a food item.')
            return redirect('home')
        return redirect('recommendations', food_name=food_name)

    return render(request, 'home.html', {
        'food_groups': food_groups,
        'food_items': food_items,
        'selected_group': selected_group,
    })


#History view - History of recommendations render to the history.html
@login_required
def history_view(request):
    history = RecommendationHistory.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'history.html', {'history': history})


@require_POST
@login_required
def update_rating(request, history_id):
    try:
        # 1. Verify the history item belongs to current user before any modifications
        history_item = RecommendationHistory.objects.get(id=history_id, user=request.user)
        
        # 2. Parse JSON payload containing rating update data
        data = json.loads(request.body)
        food_num = data.get('food_num')  # Which recommendation to update (1-3)
        rating = data.get('rating')     # New rating value
        
        # 3. Update the corresponding rating field based on food_num
        if food_num == '1':
            history_item.rating1 = rating
        elif food_num == '2':
            history_item.rating2 = rating
        elif food_num == '3':
            history_item.rating3 = rating
            
        history_item.save()
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)
    

#Autocomplete functionality for search bar
@login_required
def autocomplete(request):
    query = request.GET.get('q', '')
    # Find up to 10 food items whose names contain the query (case-insensitive) and extract their names
    food_items = FoodItem.objects.filter(food_name__icontains=query).values_list('food_name', flat=True)[:10]
    return JsonResponse(list(food_items), safe=False)


#search results as soon as the user hit enter
@login_required
def search_results(request):
    # Get the search query from the URL, default to empty string if not provided
    query = request.GET.get('q', '')
    # Search for food items where the name contains the query (case-insensitive)
    food_items = FoodItem.objects.filter(food_name__icontains=query)
    return render(request, 'search_results.html', {'food_items': food_items, 'query': query})


@login_required
def recommendations_view(request, food_name):
    selected_food = FoodItem.objects.get(food_name=food_name)
    # Get recommendations
    result = get_recommendations(food_name, selected_features=settings.SELECTED_FEATURES)
    recommendations_data = result.get('recommendations', [])
    
    if not recommendations_data or not recommendations_data[0].get('top_subsets'):
        messages.error(request, 'No valid recommendations found.')
        return redirect('home')

    # Extract top subsets
    top_subsets = recommendations_data[0]['top_subsets']
    
    # Initialize session for tracking current subset
    request.session['current_rank'] = 1
    request.session['food_name'] = food_name
    request.session['top_subsets'] = [
        {
            'subset': subset['subset'][1:],  # Exclude "Subset X" label
            'ed_score': subset['ed_score'],
            'diversity_score': subset['diversity_score'],
            'cost_score': subset['cost_score'],
            'relevance_score': subset['relevance_score']
        }
        for subset in top_subsets
    ]

    # Get the first subset (rank 1)
    current_subset = top_subsets[0]['subset'][1:]  # Exclude "Subset X" label
    recommendations = []
    
    # Fetch FoodItem objects for the recommended foods
    for name in current_subset:
        try:
            food = FoodItem.objects.get(food_name=name)
            recommendations.append(food)
        except FoodItem.DoesNotExist:
            continue

    if not recommendations:
        messages.error(request, 'No valid recommendations available.')
        return redirect('home')

    # Save to RecommendationHistory
    recommendation_history = RecommendationHistory.objects.create(
        user=request.user,
        selected_food=selected_food,
        recommended_food_1=recommendations[0] if len(recommendations) >= 1 else None,
        recommended_food_2=recommendations[1] if len(recommendations) >= 2 else None,
        recommended_food_3=recommendations[2] if len(recommendations) >= 3 else None,
    )

    return render(request, 'recommendations.html', {
        'selected_food': selected_food,
        'recommendations': recommendations,
        'recommendation_id': recommendation_history.id,
        'current_rank': request.session['current_rank'],
        'max_alternates': 4,
    })


@login_required
@csrf_exempt
def submit_rating(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=405)

    try:
        # Parse incoming JSON
        data = json.loads(request.body)
        recommendation_id = data.get('recommendation_id')
        food_id = data.get('food_id')
        rating = float(data.get('rating'))

        # Check rating is valid
        if not 0 <= rating <= 5:
            return JsonResponse({'error': 'Rating must be between 0 and 5'}, status=400)

        # Fetch recommendation and food
        recommendation = get_object_or_404(RecommendationHistory, id=recommendation_id, user=request.user)
        food = get_object_or_404(FoodItem, id=food_id)

        # Update the appropriate rating field
        if food == recommendation.recommended_food_1:
            recommendation.rating_1 = rating
        elif food == recommendation.recommended_food_2:
            recommendation.rating_2 = rating
        elif food == recommendation.recommended_food_3:
            recommendation.rating_3 = rating
        else:
            return JsonResponse({'error': 'Food not recommended'}, status=400)

        recommendation.save()
        return JsonResponse({'status': 'success'})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except FoodItem.DoesNotExist:
        return JsonResponse({'error': 'Food item not found'}, status=400)
    except RecommendationHistory.DoesNotExist:
        return JsonResponse({'error': 'Recommendation not found'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Invalid request: {str(e)}'}, status=400)


@login_required
def alternate_recommendations_view(request, food_name):
    current_rank = request.session.get('current_rank', 1)
    top_subsets = request.session.get('top_subsets', [])
    
    if current_rank >= 4 or not top_subsets:
        return JsonResponse({
            'status': 'error',
            'message': 'No more alternate suggestions available.'
        }, status=400)

    # Increment rank
    next_rank = current_rank + 1
    request.session['current_rank'] = next_rank

    # Get the next subset
    next_subset = top_subsets[next_rank - 1]['subset']  # Subsets are 0-indexed
    recommendations = []
    
    try:
        selected_food = FoodItem.objects.get(food_name=food_name)
    except FoodItem.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Selected food not found.'
        }, status=400)

    # Fetch FoodItem objects
    for name in next_subset:
        try:
            food = FoodItem.objects.get(food_name=name)
            recommendations.append({
                'id': food.id,
                'food_name': food.food_name
            })
        except FoodItem.DoesNotExist:
            continue

    if not recommendations:
        return JsonResponse({
            'status': 'error',
            'message': 'No valid recommendations available.'
        }, status=400)

    # Save to RecommendationHistory
    food_objects = [FoodItem.objects.get(id=rec['id']) for rec in recommendations]
    recommendation_history = RecommendationHistory.objects.create(
        user=request.user,
        selected_food=selected_food,
        recommended_food_1=food_objects[0] if len(food_objects) >= 1 else None,
        recommended_food_2=food_objects[1] if len(food_objects) >= 2 else None,
        recommended_food_3=food_objects[2] if len(food_objects) >= 3 else None,
    )

    return JsonResponse({
        'status': 'success',
        'recommendations': recommendations,
        'current_rank': next_rank,
        'recommendation_id': recommendation_history.id,
        'max_alternates': 4
    })
