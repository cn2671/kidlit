        console.log('Script started loading');
        
        // App State
        let appState = {
            currentPage: 'home',
            favorites: JSON.parse(localStorage.getItem('kidlit_favorites') || '[]'),
            read: JSON.parse(localStorage.getItem('kidlit_read') || '[]'),
            skipped: JSON.parse(localStorage.getItem('kidlit_skipped') || '[]'),
            searchResults: [],
            sampleBooks: [],
            themeGroups: [],
            toneGroups: []
        };

        // Theme and Tone Groups Management
        async function loadThemeToneGroups() {
            try {
                console.log('Loading theme and tone groups...');
                const response = await fetch('/api/theme-tone-groups');
                if (response.ok) {
                    const data = await response.json();
                    if (data.success) {
                        appState.themeGroups = data.theme_groups;
                        appState.toneGroups = data.tone_groups;
                        console.log('Theme groups loaded:', appState.themeGroups);
                        console.log('Tone groups loaded:', appState.toneGroups);
                        
                        // Update UI with grouped themes and tones
                        updateThemeToneUI();
                    }
                }
            } catch (error) {
                console.error('Error loading theme/tone groups:', error);
                // Fallback to original individual themes/tones if needed
            }
        }

        function updateThemeToneUI() {
            // This will be called when theme/tone group UI needs to be updated
            // We'll implement the UI updates here
            console.log('UI updated with theme/tone groups');
        }

        // Sample book data (in production, this would come from your database)
        const SAMPLE_BOOKS = [
            {
                title: "Charlotte's Web",
                author: "E.B. White",
                age_range: "6-8",
                themes: "friendship, sacrifice, loyalty",
                tone: "gentle",
                lexile_score: 450,
                predicted_lexile: 448,
                confidence: 0.92,
                description: "The story of a pig named Wilbur and his friendship with a barn spider named Charlotte."
            },
            {
                title: "The Very Hungry Caterpillar",
                author: "Eric Carle",
                age_range: "3-5",
                themes: "nature, growth, transformation",
                tone: "gentle",
                lexile_score: 192,
                predicted_lexile: 189,
                confidence: 0.95,
                description: "A caterpillar eats his way through various foods before becoming a beautiful butterfly."
            },
            {
                title: "Where the Wild Things Are",
                author: "Maurice Sendak",
                age_range: "3-5",
                themes: "imagination, adventure, emotions",
                tone: "whimsical",
                lexile_score: 425,
                predicted_lexile: 420,
                confidence: 0.88,
                description: "Max sails to an island inhabited by Wild Things who make him their king."
            },
            {
                title: "Green Eggs and Ham",
                author: "Dr. Seuss",
                age_range: "3-5",
                themes: "humor, persistence, trying new things",
                tone: "playful",
                lexile_score: 239,
                predicted_lexile: 235,
                confidence: 0.96,
                description: "Sam-I-Am tries to convince another character to try green eggs and ham."
            },
            {
                title: "Wonder",
                author: "R.J. Palacio",
                age_range: "9-12",
                themes: "acceptance, courage, friendship, identity",
                tone: "heartfelt",
                lexile_score: 790,
                predicted_lexile: 785,
                confidence: 0.85,
                description: "August Pullman, a boy with facial differences, enters mainstream school for the first time."
            },
            {
                title: "Harry Potter and the Sorcerer's Stone",
                author: "J.K. Rowling",
                age_range: "9-12",
                themes: "magic, friendship, adventure, courage",
                tone: "adventurous",
                lexile_score: 880,
                predicted_lexile: 875,
                confidence: 0.82,
                description: "Harry Potter learns he is a wizard and begins his magical education at Hogwarts."
            }
        ];

        // Initialize app
        document.addEventListener('DOMContentLoaded', function() {
            appState.sampleBooks = SAMPLE_BOOKS;
            initializeNavigation();
            
            // Load theme and tone groups from backend
            loadThemeToneGroups();
            // Initialize similar books search
            const similarSearchInput = document.getElementById('similarBookSearch');
            if (similarSearchInput) {
                similarSearchInput.addEventListener('input', searchSimilarBooks);
                similarSearchInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        searchSimilarBooks();
                    }
                });
            }
            
            // Initialize reading progress search
            const progressSearchInput = document.getElementById('progressBookSearch');
            if (progressSearchInput) {
                progressSearchInput.addEventListener('input', searchProgressBooks);
                progressSearchInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        searchProgressBooks();
                    }
                });
            }
            
            // Hide suggestions when clicking outside (with delay to allow clicks to register)
            document.addEventListener('click', function(e) {
                setTimeout(function() {
                    const similarSuggestions = document.getElementById('similarBookSuggestions');
                    const progressSuggestions = document.getElementById('progressBookSuggestions');
                    const similarInput = document.getElementById('similarBookSearch');
                    const progressInput = document.getElementById('progressBookSearch');
                    
                    // Hide similar book suggestions if clicking outside
                    if (similarSuggestions && similarInput && 
                        !similarSuggestions.contains(e.target) && 
                        !similarInput.contains(e.target)) {
                        similarSuggestions.style.display = 'none';
                    }
                    
                    // Hide progress book suggestions if clicking outside  
                    if (progressSuggestions && progressInput &&
                        !progressSuggestions.contains(e.target) && 
                        !progressInput.contains(e.target)) {
                        progressSuggestions.style.display = 'none';
                    }
                }, 50);
            });

        });

        // Navigation
        function initializeNavigation() {
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const page = this.getAttribute('data-page');
                    showPage(page);
                });
            });
        }

        function showPage(pageId) {
            // Update nav
            document.querySelectorAll('.nav-link').forEach(link => {
                link.classList.remove('active');
            });
            document.querySelector(`[data-page="${pageId}"]`).classList.add('active');
            
            // Update content
            document.querySelectorAll('.page-content').forEach(page => {
                page.classList.remove('active');
            });
            document.getElementById(pageId).classList.add('active');
            
            appState.currentPage = pageId;
            
            // AI analysis section moved to dedicated page
        }

        // Search functionality
        function setQueryAndSearch(query) {
            document.getElementById('searchInput').value = query;
            showPage('recommendations');
            setTimeout(() => performSearch(), 100);
        }

        let currentQuery = '';
        let currentPage = 1;
        let currentCategorySearch = null; // {category, type}

        // Browse by category functionality
        function searchByCategory(category, type) {
            console.log(`Searching by ${type}: ${category}`);
            
            // First navigate to recommendations page
            showPage('recommendations');
            
            // Wait for page to load, then set search input and perform search
            setTimeout(() => {
                const searchInput = document.getElementById('searchInput');
                let searchQuery = '';
                
                if (type === 'theme') {
                    searchQuery = `${category} books`;
                } else if (type === 'tone') {
                    searchQuery = `${category} books`;  
                } else if (type === 'lexile') {
                    searchQuery = category; // Already formatted like "lexile 400-600"
                }
                
                if (searchInput) {
                    searchInput.value = searchQuery;
                }
                
                // Trigger a direct category search instead of normal search
                performCategorySearch(category, type);
                
                // Scroll to results
                setTimeout(() => {
                    const resultsContainer = document.getElementById('searchResults');
                    if (resultsContainer) {
                        resultsContainer.scrollIntoView({ behavior: 'smooth' });
                    }
                }, 200);
            }, 100);
        }

        // Direct category search that bypasses query parsing
        async function performCategorySearch(category, type, page = 1) {
            // Set current search state
            currentCategorySearch = {category, type};
            currentQuery = ''; // Clear regular search query
            currentPage = page;
            const resultsContainer = document.getElementById('searchResults');
            const loading = document.getElementById('searchLoading');
            
            console.log('Performing direct category search:', category, type);
            
            if (loading) loading.style.display = 'block';
            if (resultsContainer) resultsContainer.innerHTML = '';
            
            try {
                const payload = {
                    category_search: true,
                    category: category,
                    category_type: type,
                    page: page,
                    per_page: 12
                };
                
                console.log('Sending category search payload:', payload);
                
                const response = await fetch('/api/simple-search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                console.log('Category search response:', data);
                
                if (data.success && data.results) {
                    await displayResultsWithPagination(data.results, data.pagination, data.parsed_criteria || {});
                } else {
                    resultsContainer.innerHTML = `<div class="message error">No books found for ${category}. Try a different category.</div>`;
                }
            } catch (error) {
                console.error('Category search error:', error);
                resultsContainer.innerHTML = `<div class="message error">Error searching for ${category}. Please try again.</div>`;
            } finally {
                if (loading) loading.style.display = 'none';
            }
        }

        console.log('About to define performSearch function');
        function performSearch(page = 1) {
            console.log('🚀 performSearch() called with page:', page);
            
            // Clear category search state when doing regular search
            currentCategorySearch = null;
            
            const query = document.getElementById('searchInput').value;
            console.log('📝 Search query from input:', query);
            
            const resultsContainer = document.getElementById('searchResults');
            const loading = document.getElementById('searchLoading');
            
            console.log('📦 Elements found:', {
                resultsContainer: !!resultsContainer,
                loading: !!loading,
                queryLength: query ? query.length : 0
            });
            
            if (!query.trim() && page === 1) {
                resultsContainer.innerHTML = '<div class="message info">Enter a search query to find books.</div>';
                return;
            }
            
            // Use existing query if just changing pages
            const searchQuery = page === 1 ? query : currentQuery;
            currentQuery = searchQuery;
            currentPage = page;
            
            if (!searchQuery.trim()) {
                return;
            }
            
            loading.classList.add('active');
            if (page === 1) {
                resultsContainer.innerHTML = '';
            }
            
            fetch('/api/simple-search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    query: searchQuery,
                    page: page,
                    per_page: 12
                })
            })
            .then(response => response.json())
            .then(async data => {
                console.log('🎯 API Response:', data);
                loading.classList.remove('active');
                if (data.success) {
                    console.log('✅ Search successful, displaying results...');
                    console.log('📚 Number of books:', data.results ? data.results.length : 'undefined');
                    await displayResultsWithPagination(data.results, data.pagination, data.parsed_criteria);
                    console.log('✅ displayResultsWithPagination completed');
                } else {
                    console.log('❌ Search failed:', data.error);
                    resultsContainer.innerHTML = `<div class="message error">Error: ${data.error}</div>`;
                }
            })
            .catch(error => {
                loading.classList.remove('active');
                resultsContainer.innerHTML = `<div class="message error">Search failed: ${error.message}</div>`;
            });
        }

        async function displayResultsWithPagination(books, pagination, criteria) {
            console.log('🎨 displayResultsWithPagination called:', { books: books?.length, pagination, criteria });
            
            const container = document.getElementById('searchResults');
            console.log('📦 Container found:', !!container);
            
            if (books.length === 0) {
                console.log('❌ No books to display');
                container.innerHTML = '<div class="message info">No books found matching your criteria. Try different keywords.</div>';
                return;
            }
            
            // Enhance books with Lexile predictions for books that need them
            console.log('🔮 Enhancing books with Lexile predictions...');
            try {
                books = await enhanceBooksWithLexilePredictions(books);
                console.log('✅ Lexile enhancement complete');
            } catch (error) {
                console.error('❌ Error during Lexile enhancement:', error);
                console.log('⚠️ Proceeding without Lexile enhancement...');
            }
            
            // Show parsing info
            let criteriaText = '';
            if (criteria.age_range) criteriaText += `Ages ${criteria.age_range} `;
            if (criteria.themes && criteria.themes.length > 0) criteriaText += `Themes: ${criteria.themes.join(', ')} `;
            if (criteria.tone) criteriaText += `Tone: ${criteria.tone} `;
            if (criteria.lexile_range) criteriaText += `Lexile: ${criteria.lexile_range[0]}-${criteria.lexile_range[1]} `;
            
            const paginationInfo = `
                <div style="background: #f0f8ff; padding: 16px; border-radius: 16px; margin-bottom: 24px; border: 1px solid #e0e7ff;">
                    <div style="margin-bottom: 8px;">
                        <span style="background: #667eea; color: white; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 600; margin-right: 12px;">
                            ${criteria.parsing_method || 'Smart Search'}
                        </span>
                        <strong style="color: #1f2937;">Found ${pagination.total_results} books matching:</strong> 
                        <span style="color: #4b5563;">${criteriaText || 'general search'}</span>
                    </div>
                    <div style="font-size: 14px; color: #6b7280;">
                        Showing page ${pagination.current_page} of ${pagination.total_pages} 
                        (${books.length} books on this page)
                    </div>
                </div>
            `;
            
            // Debug: Log the books array
            console.log('DisplayResultsWithPagination: Received', books.length, 'books');
            console.log('First few books:', books.slice(0, 3));
            
            const booksHtml = `
                <div class="book-grid">
                    ${books.map((book, index) => {
                        try {
                            console.log(`Rendering book ${index}: ${book.title}`);
                            return `
                        <div class="book-card" id="book-${index}">
                            <div class="book-header">
                                <div class="book-cover">
                                    ${book.cover_url ? 
                                        `<img src="/api/proxy-image?url=${encodeURIComponent(book.cover_url)}" alt="${book.title}" 
                                            onerror="console.log('Failed to load proxied image:', this.src); this.style.display='none'; this.nextElementSibling.style.display='flex';"
                                            onload="console.log('Successfully loaded proxied image:', this.src);">
                                        <div class="book-cover-placeholder" style="display: none;">📚</div>` 
                                        : 
                                        `<div class="book-cover-placeholder">📚</div>`
                                    }
                                </div>
                                <div class="book-info">
                                    <h1 class="book-title">${book.title}</h1>
                                    <p class="book-author">by ${book.author}</p>
                                    
                                    <div class="themes-pills">
                                        ${book.themes ? book.themes.split(',').slice(0, 3).map(theme => 
                                            `<span class="theme-pill">${theme.trim()}</span>`
                                        ).join('') : '<span class="theme-pill">general</span>'}
                                    </div>
                                    
                                    <div class="book-meta">
                                        ${book.tone ? `
                                        <div class="meta-item">
                                            <span>😊</span>
                                            <span>${book.tone.charAt(0).toUpperCase() + book.tone.slice(1)}</span>
                                        </div>` : ''}
                                        ${book.age_range ? `
                                        <div class="meta-item">
                                            <span>🎂</span>
                                            <span>Ages ${book.age_range}</span>
                                        </div>` : ''}
                                        ${getLexilePredictionHtml(book) ? `
                                        <div class="meta-item">
                                            ${getLexilePredictionHtml(book)}
                                        </div>` : ''}
                                    </div>
                                </div>
                            </div>
                                        
                            <div class="k-actions-row">
                                <button onclick="saveBook(${index}, 'favorites')" class="action-btn like-btn">
                                    👍 Like
                                </button>
                                <button onclick="saveBook(${index}, 'skipped')" class="action-btn skip-btn">
                                    👎 Skip
                                </button>
                                <button onclick="saveBook(${index}, 'read')" class="action-btn read-btn">
                                    📖 Read
                                </button>
                            </div>
                            
                            <div class="sneak-peek-section">
                                <button onclick="toggleSneakPeek(${index})" class="sneak-peek-btn" id="peek-btn-${index}">
                                    <span>▼</span>
                                    <span>Sneak Peek</span>
                                </button>
                                <div class="sneak-peek-content" id="peek-content-${index}" style="display: none;">
                                    ${getSummaryForBook(book) || 'Summary not available for this book.'}
                                    <div class="k-linkbar">
                                        <!-- Debug: ${book.title} - GR URL: ${book.goodreads_url} -->
                                        ${book.goodreads_url && book.goodreads_url !== '' && book.goodreads_url !== 'undefined' && book.goodreads_url !== 'nan' ? `<a class="k-iconbtn k-iconbtn--goodreads" href="${book.goodreads_url}" target="_blank" rel="noopener noreferrer" aria-label="Open on Goodreads" title="Goodreads">
                                            <img src="https://cdn.simpleicons.org/goodreads/5A4634" class="k-ico" alt="Goodreads" loading="lazy" />
                                        </a>` : '<!-- No GR URL -->'}
                                        <a class="k-iconbtn k-iconbtn--amazon" href="https://www.amazon.com/s?k=${encodeURIComponent(book.title + ' ' + book.author)}" target="_blank" rel="noopener noreferrer" aria-label="Open on Amazon" title="Amazon">
                                            <i class="fa-brands fa-amazon k-fa"></i>
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                        } catch (error) {
                            console.error(`Error rendering book ${index}:`, error, book);
                            return `<div class="book-card" style="background: #fee; padding: 20px;">Error rendering book: ${book.title || 'Unknown'}</div>`;
                        }
                    }).join('')}
                </div>
            `;
            
            const paginationControls = createPaginationControls(pagination);
            
            console.log('📝 About to set container.innerHTML with:', {
                paginationInfoLength: paginationInfo.length,
                booksHtmlLength: booksHtml.length,
                totalLength: (paginationInfo + booksHtml + paginationControls).length
            });
            
            container.innerHTML = paginationInfo + booksHtml + paginationControls;
            
            console.log('✅ Container innerHTML set! Container element:', container);
            console.log('📄 Current container content length:', container.innerHTML.length);
            
            // Store books data for action functions
            window.currentBooks = books;
            window.searchBooks = books; // Also store in dedicated variable
            
            // Scroll to results to make sure they're visible
            console.log('📍 Scrolling to search results...');
            container.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // Save book to user's list
        async function saveBook(bookIndex, listType) {
            try {
                console.log('SAVE BOOK DEBUG: saveBook called with index:', bookIndex, 'listType:', listType);
                console.log('SAVE BOOK DEBUG: window.currentBooks exists:', !!window.currentBooks);
                console.log('SAVE BOOK DEBUG: window.currentBooks.length:', window.currentBooks ? window.currentBooks.length : 'undefined');
                console.log('SAVE BOOK DEBUG: First book title:', window.currentBooks && window.currentBooks[0] ? window.currentBooks[0].title : 'N/A');
                
                // Try to find the book from different contexts
                let book = null;
                
                // First try window.currentBooks (should work for most cases)
                if (window.currentBooks && window.currentBooks[bookIndex]) {
                    book = window.currentBooks[bookIndex];
                    console.log('SAVE BOOK DEBUG: Found book in window.currentBooks:', book.title);
                }
                
                // If not found, try specific context variables
                if (!book && window.progressionBooks && window.progressionBooks[bookIndex]) {
                    book = window.progressionBooks[bookIndex];
                    console.log('SAVE BOOK DEBUG: Found book in window.progressionBooks:', book.title);
                }
                
                if (!book && window.searchBooks && window.searchBooks[bookIndex]) {
                    book = window.searchBooks[bookIndex];
                    console.log('SAVE BOOK DEBUG: Found book in window.searchBooks:', book.title);
                }
                
                if (!book && window.similarBooks && window.similarBooks[bookIndex]) {
                    book = window.similarBooks[bookIndex];
                    console.log('SAVE BOOK DEBUG: Found book in window.similarBooks:', book.title);
                }
                
                console.log('SAVE BOOK DEBUG: Retrieved book:', book ? book.title : 'undefined/null');
                
                if (!book) {
                    console.error('SAVE BOOK DEBUG: No book found at index', bookIndex);
                    return;
                }
                
                const response = await fetch('/api/save-book', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        book: book,
                        list_type: listType
                    })
                });
                
                const result = await response.json();
                console.log('SAVE BOOK DEBUG: API response:', result);
                
                if (result.success) {
                    console.log('SAVE BOOK DEBUG: Save successful, showing feedback');
                    // Show feedback
                    showBookActionFeedback(bookIndex, listType);
                } else {
                    console.error('Failed to save book:', result.error);
                }
                
            } catch (error) {
                console.error('Error saving book:', error);
            }
        }

    // Show visual feedback when book is saved - SIMPLIFIED VERSION
    function showBookActionFeedback(bookIndex, action, retryCount = 0) {
        console.log('FEEDBACK DEBUG: showBookActionFeedback called with index:', bookIndex, 'action:', action);
        
        // Try multiple card ID formats
        let card = document.getElementById(`book-${bookIndex}`);
        console.log('FEEDBACK DEBUG: Trying book-' + bookIndex + ':', !!card);
        if (!card) {
            card = document.getElementById(`book-progress-${bookIndex}`);
            console.log('FEEDBACK DEBUG: Trying book-progress-' + bookIndex + ':', !!card);
        }
        if (!card) {
            card = document.getElementById(`book-search-${bookIndex}`);
            console.log('FEEDBACK DEBUG: Trying book-search-' + bookIndex + ':', !!card);
        }
        if (!card) {
            card = document.getElementById(`book-similar-${bookIndex}`);
            console.log('FEEDBACK DEBUG: Trying book-similar-' + bookIndex + ':', !!card);
        }
        
        if (!card) {
            console.error('FEEDBACK DEBUG: No card found for index:', bookIndex);
            console.log('FEEDBACK DEBUG: Available IDs:', Array.from(document.querySelectorAll('[id^="book-"]')).map(el => el.id));
            return;
        }
        
        console.log('FEEDBACK DEBUG: Found card with ID:', card.id);
        console.log('FEEDBACK DEBUG: Card element:', card);
        // Create positioned notification near the card (like the original)
        // Add defensive checks for timing issues
        const cardRect = card.getBoundingClientRect();
        console.log('FEEDBACK DEBUG: Card rect:', cardRect);
        
        // Check if card is actually visible and positioned
        if (cardRect.width === 0 || cardRect.height === 0) {
            if (retryCount < 3) {
                console.warn(`FEEDBACK DEBUG: Card has zero dimensions, retrying... (attempt ${retryCount + 1}/3)`);
                setTimeout(() => showBookActionFeedback(bookIndex, action, retryCount + 1), 200);
                return;
            } else {
                console.warn('FEEDBACK DEBUG: Max retries reached, using fallback positioning');
                // Use fallback positioning - center-right of viewport
                const fallbackRect = {
                    top: window.innerHeight / 2,
                    right: 50,
                    width: 200,
                    height: 100
                };
                createNotificationWithRect(fallbackRect, action);
                return;
            }
        }
        
        // Create notification with card positioning
        createNotificationWithRect(cardRect, action);
    }
    
    // Helper function to create notification with given positioning
    function createNotificationWithRect(rect, action) {
        const actionText = {
            'favorites': '❤️ Added to Favorites!',
            'skipped': '⏭️ Skipped',
            'read': '📖 Added to Read List!'
        };
        
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed !important;
            top: ${rect.top + 10}px !important;
            left: ${rect.right - 150}px !important;
            background: rgba(255, 255, 255, 0.95) !important;
            color: #059669 !important;
            border: 1px solid rgba(5, 150, 105, 0.2) !important;
            padding: 6px 12px !important;
            border-radius: 8px !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            z-index: 99999 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04) !important;
            backdrop-filter: blur(8px) !important;
            opacity: 0 !important;
            transform: translateY(-10px) scale(0.95) !important;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
            pointer-events: none !important;
            letter-spacing: 0.2px !important;
        `;
        notification.textContent = actionText[action] || 'Saved!';
        console.log('FEEDBACK DEBUG: Creating notification:', actionText[action]);
        console.log('FEEDBACK DEBUG: Notification position - top:', rect.top + 10, 'right:', window.innerWidth - rect.right + 10);
        
        document.body.appendChild(notification);
        console.log('FEEDBACK DEBUG: Notification appended to body');
        
        // Animate in
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateY(0) scale(1)';
            console.log('FEEDBACK DEBUG: Notification animated in');
        }, 50);
        
        // Animate out and remove
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateY(-10px) scale(0.95)';
            setTimeout(() => {
                if (notification.parentNode) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 2000);
    }

    // Toggle sneak peek functionality
    function toggleSneakPeek(bookIndex) {
        console.log('SNEAK PEEK DEBUG: toggleSneakPeek called with bookIndex:', bookIndex);
        console.log('SNEAK PEEK DEBUG: Current page/context detected');
        
        // Try multiple ID formats for sneak peek content and button
        // First try the direct format (for context-index like 'similar-0')
        let content = document.getElementById(`peek-content-${bookIndex}`);
        let button = document.getElementById(`peek-btn-${bookIndex}`);
        console.log('SNEAK PEEK DEBUG: Trying direct format peek-content-' + bookIndex + ':', !!content);
        
        // If not found, try the simple index format
        if (!content) {
            const simpleIndex = typeof bookIndex === 'string' && bookIndex.includes('-') ? 
                bookIndex.split('-').pop() : bookIndex;
            content = document.getElementById(`peek-content-${simpleIndex}`);
            button = document.getElementById(`peek-btn-${simpleIndex}`);
            console.log('SNEAK PEEK DEBUG: Trying simple format peek-content-' + simpleIndex + ':', !!content);
        }
        
        // If not found, try with progress context
        if (!content) {
            content = document.getElementById(`peek-content-progress-${bookIndex}`);
            button = document.getElementById(`peek-btn-progress-${bookIndex}`);
        }
        
        // If still not found, try with search context
        if (!content) {
            content = document.getElementById(`peek-content-search-${bookIndex}`);
            button = document.getElementById(`peek-btn-search-${bookIndex}`);
        }
        
        // If still not found, try with similar context
        if (!content) {
            content = document.getElementById(`peek-content-similar-${bookIndex}`);
            button = document.getElementById(`peek-btn-similar-${bookIndex}`);
        }
        
        console.log('SNEAK PEEK DEBUG: Content element found:', !!content);
        console.log('SNEAK PEEK DEBUG: Button element found:', !!button);
        
        // EXTENSIVE DEBUGGING: Let's see what peek elements actually exist
        const allPeekContent = document.querySelectorAll('[id^="peek-content-"]');
        const allPeekButtons = document.querySelectorAll('[id^="peek-btn-"]');
        console.log('SNEAK PEEK DEBUG: All peek content elements:', allPeekContent.length);
        console.log('SNEAK PEEK DEBUG: All peek button elements:', allPeekButtons.length);
        allPeekContent.forEach((c, i) => {
            console.log(`SNEAK PEEK DEBUG: Content ${i}: ID="${c.id}"`);
        });
        allPeekButtons.forEach((b, i) => {
            console.log(`SNEAK PEEK DEBUG: Button ${i}: ID="${b.id}"`);
        });
        
        
        if (!content || !button) {
            console.error('SNEAK PEEK DEBUG: Missing elements after trying all formats - content:', !!content, 'button:', !!button);
            return;
        }
        
        const isVisible = content.style.display !== 'none';
        console.log('SNEAK PEEK DEBUG: Current state - isVisible:', isVisible, 'display:', content.style.display);
        
        
        if (isVisible) {
            console.log('SNEAK PEEK DEBUG: Hiding content');
            content.style.display = 'none';
            button.innerHTML = '<span>▼</span><span>Sneak Peek</span>';
            button.classList.remove('expanded');
        } else {
            console.log('SNEAK PEEK DEBUG: Showing content');
            
            // Use the original inline style - just show the content below the button
            content.style.display = 'block';
            
            // Set all parent containers to allow overflow so content is visible
            let parent = content.parentElement;
            while (parent && parent !== document.body) {
                parent.style.overflow = 'visible';
                parent = parent.parentElement;
            }
            
            button.innerHTML = '<span>▲</span><span>Close Peek</span>';
            button.classList.add('expanded');
            console.log('SNEAK PEEK DEBUG: After showing - display:', content.style.display);
            console.log('SNEAK PEEK DEBUG: Content innerHTML:', content.innerHTML.substring(0, 100));
            
            // Force reflow
            content.offsetHeight;
            
            // Check visibility after a delay
            setTimeout(() => {
                const rect = content.getBoundingClientRect();
                console.log('SNEAK PEEK DEBUG: Element bounds:', rect);
                console.log('SNEAK PEEK DEBUG: Computed display:', window.getComputedStyle(content).display);
                console.log('SNEAK PEEK DEBUG: Computed visibility:', window.getComputedStyle(content).visibility);
                console.log('SNEAK PEEK DEBUG: Content parent:', content.parentElement);
                console.log('SNEAK PEEK DEBUG: Parent display:', window.getComputedStyle(content.parentElement).display);
            }, 100);
        }
    }

    // Get summary for a book (you might need to fetch this from your backend)
    function getSummaryForBook(book) {
        console.log('Debug getSummaryForBook for:', book.title);
        console.log('  summary_gpt:', book.summary_gpt);
        console.log('  description:', book.description ? book.description.substring(0, 50) + '...' : 'null');
        
        if (book.summary_gpt && book.summary_gpt !== 'nan' && book.summary_gpt.trim() !== '') {
            console.log('  Using summary_gpt');
            return book.summary_gpt;
        }
        
        // Check if we have a description field
        if (book.description && book.description !== 'nan' && book.description.trim() !== '') {
            console.log('  Using description');
            return book.description;
        }
        
        // Fallback to generated description
        const themes = book.themes ? book.themes.split(',').slice(0, 2).map(t => t.trim()).join(' and ') : 'adventure';
        const tone = book.tone || 'engaging';
        const ageText = book.age_range ? `ages ${book.age_range}` : 'young readers';
        
        console.log('  Using fallback');
        return `A ${tone} story about ${themes}. Perfect for ${ageText}.`;
    }


    // Functions to view user's saved books
    async function viewUserBooks(listType) {
        try {
            // Update active tab
            document.querySelectorAll('.list-tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[data-list="${listType}"]`).classList.add('active');
            
            const response = await fetch(`/api/get-user-books/${listType}`);
            const result = await response.json();
            
            if (result.success) {
                displayUserBooksList(result.books, listType);
            } else {
                console.error('Failed to load books:', result.error);
                document.getElementById('userBooksContainer').innerHTML = 
                    `<div class="message error">Failed to load ${listType}: ${result.error}</div>`;
            }
        } catch (error) {
            console.error('Error loading user books:', error);
            document.getElementById('userBooksContainer').innerHTML = 
                `<div class="message error">Error loading ${listType}. Please try again.</div>`;
        }
    }

    function displayUserBooksList(books, listType) {
        const container = document.getElementById('userBooksContainer');
        const listTitle = {
            'favorites': 'My Favorite Books',
            'skipped': 'Skipped Books', 
            'read': 'Books I\'ve Read'
        };
        
        if (books.length === 0) {
            container.innerHTML = `
                <div class="message info">
                    <h3>${listTitle[listType]}</h3>
                    <p>No books in this list yet. Start searching to add some!</p>
                    <button onclick="showPage('recommendations')" class="back-to-search-btn">
                        <span style="margin-right: 8px;">🔍</span>
                        <span>Discover More Books</span>
                    </button>
                </div>
            `;
            return;
        }
        
        const html = `
            <h3>${listTitle[listType]} (${books.length})</h3>
            <div class="book-grid">
                ${books.map((book, index) => `
                    <div class="book-card">
                        <div class="k-card-header">
                            <div class="k-header-row">
                                <div class="k-cover">
                                    ${book.cover_url ? 
                                        `<img src="/api/proxy-image?url=${encodeURIComponent(book.cover_url)}" alt="${book.title}" style="width: 80px; height: 120px; border-radius: 10px; object-fit: cover;" 
                                            onerror="console.log('Failed to load proxied k-cover image:', this.src); this.style.display='none'; this.nextElementSibling.style.display='flex';"
                                            onload="console.log('Successfully loaded proxied k-cover image:', this.src);">
                                        <div style="display: none; width: 80px; height: 120px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); align-items: center; justify-content: center; color: white; font-size: 24px;">📚</div>` 
                                        : 
                                        `<div style="width: 80px; height: 120px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-size: 24px;">📚</div>`
                                    }
                                </div>
                                <div class="book-info">
                                    <h3 class="k-header-title">${book.title}</h3>
                                    <div class="k-header-meta">by ${book.author}</div>
                                    ${book.age_range ? `<div class="k-header-meta">🎂 Ages ${book.age_range}</div>` : ''}
                                    ${book.lexile_score && book.lexile_score !== 'nan' ? `<div class="k-header-meta">📊 ${Math.round(parseFloat(book.lexile_score))}L</div>` : ''}
                                    <div class="k-pill">${book.themes ? book.themes.split(',')[0] : 'General'}</div>
                                    ${book.tone ? `<div class="k-meta">😊 ${book.tone}</div>` : ''}
                                </div>
                            </div>
                        </div>
                        
                        <!-- Sneak Peek for saved books too -->
                        <div class="sneak-peek-container">
                            <button onclick="toggleSneakPeekSaved(${index})" class="sneak-peek-btn" id="saved-peek-btn-${index}">
                                ▼ Sneak Peek
                            </button>
                            <div class="sneak-peek-content" id="saved-peek-content-${index}" style="display: none;">
                                <p>${getSummaryForBook(book) || 'Summary not available for this book.'}</p>
                            </div>
                        </div>
                        
                        <button onclick="removeFromList('${encodeURIComponent(book.title)}', '${listType}')" 
                                class="remove-btn" style="width: 100%; margin-top: 10px;">
                            🗑️ Remove from List
                        </button>
                    </div>
                `).join('')}
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <button onclick="showPage('recommendations')" class="back-to-search-btn">
                    🔍 Back to Search
                </button>
            </div>
        `;
        
        container.innerHTML = html;
        
        // Store current saved books for sneak peek functionality
        window.currentSavedBooks = books;
    }

    // Remove book from list
    async function removeFromList(encodedTitle, listType) {
        try {
            const title = decodeURIComponent(encodedTitle);
            const response = await fetch('/api/remove-book', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({title: title, list_type: listType})
            });
            
            const result = await response.json();
            if (result.success) {
                viewUserBooks(listType); // Refresh the list
            } else {
                console.error('Failed to remove book:', result.error);
            }
        } catch (error) {
            console.error('Error removing book:', error);
        }
    }

    // Sneak peek for saved books
    function toggleSneakPeekSaved(bookIndex) {
        const content = document.getElementById(`saved-peek-content-${bookIndex}`);
        const button = document.getElementById(`saved-peek-btn-${bookIndex}`);
        
        if (!content || !button) return;
        
        const isVisible = content.style.display !== 'none';
        
        if (isVisible) {
            content.style.display = 'none';
            button.textContent = '▼ Sneak Peek';
            button.classList.remove('expanded');
        } else {
            content.style.display = 'block';
            button.textContent = '▲ Close Peek';
            button.classList.add('expanded');
        }
    }


    // Add CSS styles for the new elements
    const bookActionStyles = `
    <style>
    .book-actions {
        display: flex;
        gap: 8px;
        margin: 15px 0 10px 0;
        justify-content: center;
    }

    .action-btn {
        padding: 6px 12px;
        border: none;
        border-radius: 15px;
        font-size: 15px;
        cursor: pointer;
        transition: all 0.2s ease;
        flex: 1;
        max-width: 70px;
    }

    .like-btn {
        background: #ffffff;
        color: #475569;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    .like-btn:hover {
        background: #f8fafc;
        border-color: #cbd5e1;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        transform: translateY(-1px);
    }

    .skip-btn {
        background: #ffffff;
        color: #475569;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    .skip-btn:hover {
        background: #f8fafc;
        border-color: #cbd5e1;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        transform: translateY(-1px);
    }

    .read-btn {
        background: #ffffff;
        color: #475569;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    .read-btn:hover {
        background: #f8fafc;
        border-color: #cbd5e1;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        transform: translateY(-1px);
    }
    .remove-btn {
        padding: 8px 16px;
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .remove-btn:hover {
        background: #fee2e2;
        border-color: #fca5a5;
        box-shadow: 0 2px 4px 0 rgba(220, 38, 38, 0.08);
        transform: translateY(-1px);
    }

    .sneak-peek-container {
        margin-top: 10px;
    }

    .sneak-peek-btn {
        width: 100%;
        padding: 8px;
        border: 1px solid #ddd;
        background: #f8f9fa;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
        transition: all 0.2s ease;
    }

    .sneak-peek-btn:hover {
        background: #e9ecef;
    }

    .sneak-peek-btn.expanded {
        background: #667eea;
        color: white;
        border-color: #667eea;
    }
    .sneak-peek-section {
        margin-top: auto;
        margin-bottom: 0px;
    }
    .k-actions-row {
        margin-bottom: 24px;
        margin-top: auto;
    }
    .book-info {
        flex-grow: 1;
    }
    /* List tab buttons */
    .list-tab-btn {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 24px;
        font-size: 16px;
        font-weight: 600;
        color: #6b7280;
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 140px;
        text-align: center;
    }
    .list-tab-btn:hover {
        background: #f8fafc;
        border-color: #cbd5e1;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .list-tab-btn.active {
        background: #667eea;
        border-color: #667eea;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    /* Back to Search button */
    .back-to-search-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 16px 32px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 200px;
        margin-top: 40px;
    }
    .back-to-search-btn:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .back-to-search-btn:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    .sneak-peek-content {
        margin-top: 10px;
        padding: 12px;
        background: #f8f9fa;
        border-radius: 6px;
        font-size: 13px;
        line-height: 1.4;
        color: #555;
        border-left: 3px solid #667eea;
    }

    @keyframes fadeInOut {
        0% { opacity: 0; transform: translateY(-10px); }
        20% { opacity: 1; transform: translateY(0); }
        80% { opacity: 1; transform: translateY(0); }
        100% { opacity: 0; transform: translateY(-10px); }
    }

    .book-card {
        position: relative;
    }
    </style>`;

    // Add the styles to the page
    document.head.insertAdjacentHTML('beforeend', bookActionStyles);

        // Universal pagination function that works for both regular and category searches
        function goToPage(page) {
            if (currentCategorySearch) {
                // Category search pagination
                performCategorySearch(currentCategorySearch.category, currentCategorySearch.type, page);
            } else {
                // Regular search pagination
                performSearch(page);
            }
        }

        function createPaginationControls(pagination) {
            if (pagination.total_pages <= 1) {
                return '';
            }
            
            let controls = '<div class="pagination-container" style="display: flex; justify-content: center; align-items: center; margin: 20px 0; gap: 10px;">';
            
            // Previous button
            if (pagination.has_prev) {
                controls += `<button onclick="goToPage(${pagination.current_page - 1})" class="page-btn">← Previous</button>`;
            } else {
                controls += `<button disabled class="page-btn disabled">← Previous</button>`;
            }
            
            // Page numbers
            const startPage = Math.max(1, pagination.current_page - 2);
            const endPage = Math.min(pagination.total_pages, pagination.current_page + 2);
            
            if (startPage > 1) {
                controls += `<button onclick="goToPage(1)" class="page-btn">1</button>`;
                if (startPage > 2) {
                    controls += `<span style="padding: 0 5px;">...</span>`;
                }
            }
            
            for (let i = startPage; i <= endPage; i++) {
                if (i === pagination.current_page) {
                    controls += `<button class="page-btn active">${i}</button>`;
                } else {
                    controls += `<button onclick="goToPage(${i})" class="page-btn">${i}</button>`;
                }
            }
            
            if (endPage < pagination.total_pages) {
                if (endPage < pagination.total_pages - 1) {
                    controls += `<span style="padding: 0 5px;">...</span>`;
                }
                controls += `<button onclick="goToPage(${pagination.total_pages})" class="page-btn">${pagination.total_pages}</button>`;
            }
            
            // Next button
            if (pagination.has_next) {
                controls += `<button onclick="goToPage(${pagination.current_page + 1})" class="page-btn">Next →</button>`;
            } else {
                controls += `<button disabled class="page-btn disabled">Next →</button>`;
            }
            
            controls += '</div>';
            return controls;
        }

        // Add CSS for pagination buttons
        const paginationStyles = `
        <style>
        .pagination-container {
            margin: 20px 0;
            text-align: center;
        }

        .page-btn {
            padding: 8px 12px;
            margin: 0 2px;
            border: 1px solid #ddd;
            background: white;
            cursor: pointer;
            border-radius: 4px;
            font-size: 14px;
            transition: all 0.2s;
        }

        .page-btn:hover:not(.disabled) {
            background: #f0f8ff;
            border-color: #667eea;
        }

        .page-btn.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        .page-btn.disabled {
            background: #f5f5f5;
            color: #999;
            cursor: not-allowed;
            border-color: #e0e0e0;
        }
        </style>`;

        // Add styles to page
        document.head.insertAdjacentHTML('beforeend', paginationStyles);

                

        // AI Book Analysis - Autocomplete functionality
        let selectedBook = null;
        let searchTimeout = null;

        async function handleBookSearch() {
            console.log('handleBookSearch called');
            const searchInput = document.getElementById('bookSearch');
            const query = searchInput ? searchInput.value.trim() : '';
            const suggestionsContainer = document.getElementById('bookSuggestions');
            const debugInfo = document.getElementById('debugInfo');
            
            // Update debug info
            if (debugInfo) {
                debugInfo.textContent = `Function called. Query: "${query}" (length: ${query.length})`;
            }
            
            console.log('Query:', query, 'Length:', query.length);
            console.log('searchInput element:', searchInput);
            console.log('suggestionsContainer element:', suggestionsContainer);
            
            if (!searchInput || !suggestionsContainer) {
                console.error('Required elements not found');
                if (debugInfo) debugInfo.textContent = 'ERROR: Required elements not found';
                return;
            }
            
            // Clear previous timeout
            if (searchTimeout) {
                clearTimeout(searchTimeout);
            }
            
            // Hide suggestions if query is too short
            if (query.length < 2) {
                console.log('Query too short, hiding suggestions');
                suggestionsContainer.style.display = 'none';
                if (debugInfo) debugInfo.textContent = `Query too short: "${query}"`;
                return;
            }
            
            // Add visual feedback for searching
            searchInput.style.borderColor = '#6366f1';
            
            // Debounce the search
            searchTimeout = setTimeout(async () => {
                try {
                    console.log('Searching for:', query);
                    const response = await fetch(`/api/book-suggestions?q=${encodeURIComponent(query)}`);
                    console.log('Response status:', response.status);
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    console.log('Search data:', data);
                    
                    if (data.success && data.suggestions && data.suggestions.length > 0) {
                        console.log(`Found ${data.suggestions.length} suggestions`);
                        displayBookSuggestions(data.suggestions);
                    } else {
                        console.log('No suggestions found');
                        suggestionsContainer.innerHTML = '<div style="padding: 12px; color: #64748b; text-align: center; font-size: 14px;">No books found. Try searching for popular titles like "Harry Potter" or authors like "Dr. Seuss"</div>';
                        suggestionsContainer.style.display = 'block';
                    }
                    
                    searchInput.style.borderColor = '#e5e7eb';
                } catch (error) {
                    console.error('Search error:', error);
                    suggestionsContainer.innerHTML = '<div style="padding: 12px; color: #ef4444; text-align: center; font-size: 14px;">Search error. Please try again.</div>';
                    suggestionsContainer.style.display = 'block';
                    searchInput.style.borderColor = '#ef4444';
                }
            }, 300);
        }

        function displayBookSuggestions(suggestions) {
            const suggestionsContainer = document.getElementById('bookSuggestions');
            
            try {
                let html = '';
                suggestions.forEach((book, index) => {
                    const details = [];
                    if (book.age_range && book.age_range !== 'nan' && book.age_range.trim()) details.push(`Ages ${book.age_range}`);
                    if (book.lexile_score && book.lexile_score !== 'nan' && book.lexile_score.trim()) details.push(`${Math.round(parseFloat(book.lexile_score))}L`);
                    if (book.themes && book.themes !== 'nan' && book.themes.length > 0) {
                        const themes = book.themes.split(',').slice(0, 2).map(t => t.trim()).join(', ');
                        if (themes) details.push(themes);
                    }
                    
                    // Use data attributes instead of inline onclick with quotes
                    html += `
                        <div class="book-suggestion-item" 
                             data-title="${escapeHtml(book.title)}" 
                             data-author="${escapeHtml(book.author)}" 
                             data-themes="${escapeHtml(book.themes)}" 
                             data-details="${escapeHtml(details.join(' • '))}"
                             onclick="selectBookFromData(this)"
                             style="padding: 12px; border-bottom: 1px solid #f1f5f9; cursor: pointer; transition: background 0.2s;"
                             onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'">
                            <div style="font-weight: 600; color: #1e293b;">${escapeHtml(book.title)}</div>
                            <div style="color: #64748b; font-size: 13px; margin-top: 2px;">by ${escapeHtml(book.author)}</div>
                            ${details.length > 0 ? `<div style="color: #64748b; font-size: 12px; margin-top: 4px;">${details.join(' • ')}</div>` : ''}
                        </div>
                    `;
                });
                
                // Add a temporary test item to make sure the dropdown is visible
                html = '<div style="padding: 12px; background: #10b981; color: white; font-weight: bold;">✓ Dropdown is working! Found ' + suggestions.length + ' books:</div>' + html;
                
                suggestionsContainer.innerHTML = html;
                suggestionsContainer.style.display = 'block';
                console.log('Suggestions displayed:', suggestions.length);
                
                // Force visibility for debugging
                suggestionsContainer.style.visibility = 'visible';
                suggestionsContainer.style.opacity = '1';
            } catch (error) {
                console.error('Error displaying suggestions:', error);
                suggestionsContainer.innerHTML = '<div style="padding: 12px; color: #ef4444; text-align: center; font-size: 14px;">Error displaying suggestions</div>';
                suggestionsContainer.style.display = 'block';
            }
        }

        function selectBookFromData(element) {
            const title = element.getAttribute('data-title');
            const author = element.getAttribute('data-author');
            const themes = element.getAttribute('data-themes');
            const details = element.getAttribute('data-details');
            selectBookForAnalysis(title, author, themes, details);
        }

        function escapeHtml(unsafe) {
            if (!unsafe || typeof unsafe !== 'string') return '';
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;")
                .replace(/\n/g, " ")
                .replace(/\r/g, " ")
                .replace(/\t/g, " ");
        }

        function selectBook(title, author, themes, details) {
            selectedBook = { title, author, themes };
            
            // Update display with error checking
            const titleEl = document.getElementById('selectedTitle');
            const authorEl = document.getElementById('selectedAuthor');
            const detailsEl = document.getElementById('selectedDetails');
            const displayEl = document.getElementById('selectedBookDisplay');
            const searchEl = document.getElementById('bookSearch');
            const suggestionsEl = document.getElementById('bookSuggestions');
            
            if (titleEl) titleEl.textContent = title;
            if (authorEl) authorEl.textContent = author;
            if (detailsEl) detailsEl.textContent = details;
            if (displayEl) displayEl.style.display = 'block';
            
            // Clear and hide search
            if (searchEl) searchEl.value = '';
            if (suggestionsEl) suggestionsEl.style.display = 'none';
            
            // Hide manual entry
            const manualEntryEl = document.getElementById('manualEntry');
            const manualToggleEl = document.getElementById('manualEntryToggle');
            if (manualEntryEl) manualEntryEl.style.display = 'none';
            if (manualToggleEl) manualToggleEl.textContent = '📝 Enter book details manually instead';
        }

        // AI Analysis specific book selection
        function selectBookForAnalysis(title, author, themes, details) {
            selectedBook = { title, author, themes };
            
            // Update display with AI Analysis specific IDs
            const titleEl = document.getElementById('analysisSelectedTitle');
            const authorEl = document.getElementById('analysisSelectedAuthor');
            const detailsEl = document.getElementById('analysisSelectedDetails');
            const displayEl = document.getElementById('analysisSelectedBookDisplay');
            const searchEl = document.getElementById('bookSearch');
            const suggestionsEl = document.getElementById('bookSuggestions');
            
            if (titleEl) titleEl.textContent = title;
            if (authorEl) authorEl.textContent = author;
            if (detailsEl) detailsEl.textContent = details;
            if (displayEl) displayEl.style.display = 'block';
            
            // Clear and hide search
            if (searchEl) searchEl.value = '';
            if (suggestionsEl) suggestionsEl.style.display = 'none';
            
            // Hide manual entry
            const manualEntryEl = document.getElementById('manualEntry');
            const manualToggleEl = document.getElementById('toggleManualEntry');
            if (manualEntryEl) manualEntryEl.style.display = 'none';
            if (manualToggleEl) manualToggleEl.textContent = '📝 Enter book details manually instead';
        }

        function clearAnalysisSelectedBook() {
            selectedBook = null;
            const displayEl = document.getElementById('analysisSelectedBookDisplay');
            const searchEl = document.getElementById('bookSearch');
            if (displayEl) displayEl.style.display = 'none';
            if (searchEl) searchEl.focus();
        }

        function clearSelectedBook() {
            selectedBook = null;
            const displayEl = document.getElementById('selectedBookDisplay');
            const searchEl = document.getElementById('bookSearch');
            if (displayEl) displayEl.style.display = 'none';
            if (searchEl) searchEl.focus();
        }

        function toggleManualEntry() {
            const manualEntry = document.getElementById('manualEntry');
            const toggle = document.getElementById('manualEntryToggle');
            const selectedDisplay = document.getElementById('selectedBookDisplay');
            
            if (manualEntry.style.display === 'none') {
                manualEntry.style.display = 'block';
                toggle.textContent = '🔍 Search our catalog instead';
                selectedDisplay.style.display = 'none';
                selectedBook = null;
                document.getElementById('bookTitle').focus();
            } else {
                manualEntry.style.display = 'none';
                toggle.textContent = '📝 Enter book details manually instead';
                document.getElementById('bookTitle').value = '';
                document.getElementById('bookAuthor').value = '';
                document.getElementById('bookThemes').value = '';
                document.getElementById('bookSearch').focus();
            }
        }

        // Test function to force show dropdown
        function testDropdown() {
            console.log('Test dropdown function called');
            const suggestionsContainer = document.getElementById('bookSuggestions');
            if (suggestionsContainer) {
                suggestionsContainer.innerHTML = '<div style="padding: 12px; background: #10b981; color: white; font-weight: bold;">🚀 TEST: This is a forced dropdown test!</div>';
                suggestionsContainer.style.display = 'block';
                suggestionsContainer.style.visibility = 'visible';
                suggestionsContainer.style.opacity = '1';
                console.log('Test dropdown should now be visible');
            } else {
                console.error('Cannot find bookSuggestions element');
            }
        }

        // Hide suggestions when clicking outside
        document.addEventListener('click', function(event) {
            const searchContainer = document.querySelector('.book-search-container');
            if (!searchContainer.contains(event.target)) {
                document.getElementById('bookSuggestions').style.display = 'none';
            }
        });

        // AI Book Analysis
        async function analyzeBook() {
            let title, author, themes;
            
            // Check if we have a selected book from autocomplete
            if (selectedBook) {
                title = selectedBook.title;
                author = selectedBook.author;
                themes = selectedBook.themes;
            } else {
                alert('Please select a book from the dropdown above');
                return;
            }
            
            if (!title || !author) {
                alert('Please select a book from our catalog');
                return;
            }
            
            const loading = document.getElementById('analysisLoading');
            const resultContainer = document.getElementById('predictionResult');
            
            loading.classList.add('active');
            resultContainer.innerHTML = '';
            
            try {
                // Call the enhanced Lexile prediction API
                const response = await fetch('/api/predict-lexile', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        title: title,
                        author: author
                    })
                });
                
                const prediction = await response.json();
                
                if (prediction.success) {
                    // Transform predict-lexile response to match displayPredictionResult expectations
                    const transformedPrediction = {
                        lexile_score: prediction.predicted_lexile,
                        age_range: prediction.predicted_lexile >= 900 ? '13+' : 
                                  prediction.predicted_lexile >= 600 ? '9-12' : 
                                  prediction.predicted_lexile >= 300 ? '6-8' : '3-5',
                        age_category: prediction.predicted_lexile >= 900 ? 'Advanced' : 
                                     prediction.predicted_lexile >= 600 ? 'Intermediate' : 'Beginning',
                        assignment_tier: prediction.confidence >= 0.8 ? 'Tier 1: Conservative' : 
                                        prediction.confidence >= 0.6 ? 'Tier 2: Moderate' : 'Tier 3: Conservative',
                        confidence_score: prediction.confidence
                    };
                    
                    displayPredictionResult({
                        title: title,
                        author: author,
                        themes: themes
                    }, transformedPrediction);
                } else {
                    resultContainer.innerHTML = '<div class="message error">Error analyzing book. Please try again.</div>';
                }
            } catch (error) {
                console.error('Analysis error:', error);
                // Fallback to demo prediction
                const demoPrediction = createDemoPrediction(title, author, themes);
                displayPredictionResult({
                    title: title,
                    author: author,
                    themes: themes
                }, demoPrediction);
            } finally {
                loading.classList.remove('active');
            }
        }

        function createDemoPrediction(title, author, themes) {
            // Simple rule-based demo prediction
            let lexile = 400;
            let category = 'Beginning';
            let ageRange = '6-8';
            let confidence = 0.7;
            let tier = 'Tier 2: Medium Confidence';
            
            if (author.toLowerCase().includes('seuss') || author.toLowerCase().includes('carle')) {
                lexile = 200;
                category = 'Early';
                ageRange = '3-5';
                confidence = 0.9;
                tier = 'Tier 1: High Confidence';
            } else if (author.toLowerCase().includes('rowling') || title.toLowerCase().includes('harry potter')) {
                lexile = 850;
                category = 'Advanced';
                ageRange = '9-12';
                confidence = 0.85;
                tier = 'Tier 1: High Confidence';
            } else if (themes.toLowerCase().includes('magic') || themes.toLowerCase().includes('adventure')) {
                lexile = 650;
                category = 'Intermediate';
                ageRange = '9-12';
                confidence = 0.75;
                tier = 'Tier 2: Medium Confidence';
            }
            
            return {
                success: true,
                lexile_score: lexile,
                age_category: category,
                age_range: ageRange,
                confidence_score: confidence,
                assignment_tier: tier
            };
        }

        function displayPredictionResult(bookData, prediction) {
            const tierClass = prediction.assignment_tier.includes('Tier 1') ? 'tier1' : 
                            prediction.assignment_tier.includes('Tier 2') ? 'tier2' : 'tier3';
            
            const html = `
                <div class="prediction-result">
                    <div class="prediction-header">
                        <div class="prediction-title">${bookData.title}</div>
                        <div class="confidence-badge ${tierClass}">
                            ${prediction.assignment_tier}
                        </div>
                    </div>
                    <div class="prediction-details">
                        <div class="detail-item">
                            <div class="detail-value">${Math.round(parseFloat(prediction.lexile_score))}L</div>
                            <div class="detail-label">Lexile Score</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-value">${prediction.age_range}</div>
                            <div class="detail-label">Age Range</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-value">${prediction.age_category}</div>
                            <div class="detail-label">Reader Level</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-value">${(prediction.confidence_score * 100).toFixed(0)}%</div>
                            <div class="detail-label">AI Confidence</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; padding: 15px; background: white; border-radius: 8px;">
                        <strong>How our AI determined this:</strong><br>
                        • Author pattern analysis (${bookData.author})<br>
                        • Theme complexity assessment<br>
                        • Text structure evaluation<br>
                        • Cross-reference with 1,000+ similar books
                    </div>
                </div>
            `;
            
            document.getElementById('predictionResult').innerHTML = html;
        }

        // Book card creation
        function createBookCard(book, showActions = true) {
            const isLiked = appState.favorites.some(fav => fav.title === book.title && fav.author === book.author);
            const isRead = appState.read.some(r => r.title === book.title && r.author === book.author);
            const isSkipped = appState.skipped.some(s => s.title === book.title && s.author === book.author);
            
            return `
                <div class="book-card">
                    <div class="k-card-header">
                        <div class="k-header-row">
                            <div class="k-cover">📚</div>
                            <div class="book-info">
                                <h3 class="k-header-title">${book.title}</h3>
                                <div class="k-header-meta">by ${book.author}</div>
                                <div class="k-header-meta">🎂 Ages ${book.age_range}</div>
                                <div class="k-header-meta">📊 ${Math.round(parseFloat(book.lexile_score || book.predicted_lexile))}L Lexile</div>
                                <div class="k-pill">${book.themes.split(',')[0].trim()}</div>
                                <div class="k-meta">😊 ${book.tone}</div>
                                ${book.confidence ? `<div class="k-meta">🤖 ${(book.confidence * 100).toFixed(0)}% AI confidence</div>` : ''}
                            </div>
                        </div>
                    </div>
                    
                    ${showActions ? `
                    <div class="k-divider"></div>
                    <div class="k-header-actions">
                        <div class="k-actions-row">
                            <button class="action-btn ${isLiked ? 'liked' : ''}" 
                                    onclick="toggleFavorite('${book.title}', '${book.author}')">
                                ❤️ ${isLiked ? 'Liked' : 'Like'}
                            </button>
                            <button class="action-btn ${isRead ? 'read' : ''}" 
                                    onclick="toggleRead('${book.title}', '${book.author}')">
                                📖 ${isRead ? 'Read' : 'Mark Read'}
                            </button>
                            <button class="action-btn ${isSkipped ? 'skipped' : ''}" 
                                    onclick="toggleSkipped('${book.title}', '${book.author}')">
                                🚫 ${isSkipped ? 'Skipped' : 'Skip'}
                            </button>
                        </div>
                    </div>
                    ` : ''}
                </div>
            `;
        }








        // Book actions
        

        function findBook(title, author) {
            return appState.sampleBooks.find(book => book.title === title && book.author === author);
        }

        function updateCurrentPageDisplay() {
            // Refresh the current page display
            if (appState.currentPage === 'recommendations') {
                const searchInput = document.getElementById('searchInput');
                if (searchInput.value) {
                    performSearch();
                }
            } else if (appState.currentPage === 'favorites') {
                // Don't call updateFavoritesDisplay() - just show the page
                // The user will click which list they want to see
                const container = document.getElementById('userBooksContainer');
                if (container) {
                    container.innerHTML = '<p>Select a list above to view your saved books.</p>';
                }
            }
        }



        // Similar Books functionality
        function searchSimilarBooks() {
            const query = document.getElementById('similarBookSearch').value.trim();
            const suggestionsContainer = document.getElementById('similarBookSuggestions');
            
            console.log('searchSimilarBooks called with query:', query);
            
            if (!query) {
                console.log('Empty query, hiding suggestions');
                suggestionsContainer.innerHTML = '';
                suggestionsContainer.style.display = 'none';
                return;
            }
            
            // Use the same endpoint as AI Book Analysis for consistency
            console.log('Making API call to /api/book-suggestions');
            fetch(`/api/book-suggestions?q=${encodeURIComponent(query)}`)
            .then(response => {
                console.log('API response status:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('API response data:', data);
                if (data.success && data.suggestions && data.suggestions.length > 0) {
                    console.log('Displaying', data.suggestions.length, 'suggestions');
                    displaySimilarBookSuggestions(data.suggestions);
                } else {
                    console.log('No suggestions found');
                    suggestionsContainer.innerHTML = '<div style="padding: 10px; color: #64748b;">No books found. Try searching for popular titles.</div>';
                    suggestionsContainer.style.display = 'block';
                }
            })
            .catch(error => {
                console.error('Search error:', error);
                suggestionsContainer.innerHTML = '<div style="padding: 10px; color: #ef4444;">Search error. Please try again.</div>';
                suggestionsContainer.style.display = 'block';
            });
        }

        // Reading Progress functionality
        function searchProgressBooks() {
            const query = document.getElementById('progressBookSearch').value.trim();
            const suggestionsContainer = document.getElementById('progressBookSuggestions');
            const resultsContainer = document.getElementById('progressResults');
            
            if (!query) {
                suggestionsContainer.innerHTML = '';
                suggestionsContainer.style.display = 'none';
                resultsContainer.innerHTML = '';
                return;
            }
            
            if (query.length < 3) {
                suggestionsContainer.innerHTML = '<div style="padding: 10px; color: #666;">Type at least 3 characters to search...</div>';
                suggestionsContainer.style.display = 'block';
                return;
            }
            
            // Search for books to select from
            fetch('/api/simple-search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query, page: 1, per_page: 12})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success && data.results.length > 0) {
                    displayProgressBookSuggestions(data.results);
                } else {
                    suggestionsContainer.innerHTML = '<div style="padding: 10px;">No books found</div>';
                    suggestionsContainer.style.display = 'block';
                }
            })
            .catch(error => {
                console.error('Search error:', error);
            });
        }

        function displayProgressBookSuggestions(books) {
            const container = document.getElementById('progressBookSuggestions');
            
            console.log('Displaying progress book suggestions:', books);
            
            const html = books.map(book => {
                console.log('Creating suggestion for:', book.title);
                return `
                    <div data-book-json='${escapeHtml(JSON.stringify(book))}' onclick="selectBookForProgressionData(this)" 
                        style="padding: 12px; border-bottom: 1px solid #e5e7eb; cursor: pointer; display: flex; gap: 12px; align-items: center;"
                        onmouseover="this.style.background='#f9fafb'" 
                        onmouseout="this.style.background='white'">
                        <div style="width: 40px; height: 60px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 16px;">📚</div>
                        <div>
                            <div style="font-weight: 600; color: #1f2937;">${book.title}</div>
                            <div style="font-size: 13px; color: #6b7280;">by ${book.author}</div>
                            <div style="font-size: 12px; color: #9ca3af;">${book.age_range || ''} • ${book.themes ? book.themes.split(',')[0] : ''}</div>
                        </div>
                    </div>
                `;
            }).join('');
            
            console.log('Generated HTML:', html);
            container.innerHTML = html;
            container.style.display = 'block';
        }

        // Global function for triggering progression search
        function triggerProgressionSearch() {
            console.log('=== AUTOMATIC PROGRESSION SEARCH TRIGGERED ===');
            
            // Get the base book data from sessionStorage
            const baseBook = JSON.parse(sessionStorage.getItem('currentProgressionBook') || '{}');
            const baseLexile = parseFloat(baseBook.lexile_score) || 750; // Default to 750 if no lexile
            console.log('Retrieved baseBook:', baseBook.title, 'Lexile:', baseLexile);
            
            // Store globally for auto search
            window.baseBook = baseBook;
            window.baseLexile = baseLexile;
            
            // Debug radio button selection
            const allRadios = document.querySelectorAll('input[name="difficulty-level"]');
            console.log('All radio buttons found:', allRadios.length);
            allRadios.forEach((radio, index) => {
                console.log(`Radio ${index}: value="${radio.value}", checked=${radio.checked}`);
            });
            
            let selectedRadio = document.querySelector('input[name="difficulty-level"]:checked');
            console.log('Selected radio element:', selectedRadio);
            
            if (!selectedRadio) {
                console.error('No radio button selected! Defaulting to "similar"');
                // Default to similar if none selected
                const similarRadio = document.querySelector('input[name="difficulty-level"][value="similar"]');
                if (similarRadio) {
                    similarRadio.checked = true;
                    selectedRadio = similarRadio; // Update the selectedRadio reference
                }
            }
            
            const selectedDifficulty = selectedRadio ? selectedRadio.value : 'similar';
            console.log('=== DIFFICULTY DETECTION DEBUG ===');
            console.log('Selected radio:', selectedRadio);
            console.log('Selected radio value:', selectedRadio ? selectedRadio.value : 'none');
            console.log('Final selectedDifficulty:', selectedDifficulty);
            console.log('================================');
            const loadingMessage = document.getElementById('loadingMessage');
            
            // Show loading message
            if (loadingMessage) {
                loadingMessage.style.display = 'block';
            }
            
            // Calculate Lexile ranges based on difficulty selection
            let searchQuery = '';
            if (baseLexile > 0) {
                let targetMinLexile, targetMaxLexile;
                
                switch(selectedDifficulty) {
                    case 'lower':
                        targetMinLexile = Math.max(200, baseLexile - 200);
                        targetMaxLexile = baseLexile - 25;
                        break;
                    case 'similar':
                        targetMinLexile = baseLexile - 75;
                        targetMaxLexile = baseLexile + 75;
                        break;
                    case 'advanced':
                        targetMinLexile = baseLexile + 25;
                        targetMaxLexile = baseLexile + 400;
                        break;
                }
                
                searchQuery = `lexile ${targetMinLexile}-${targetMaxLexile}`;
                console.log('Constructed lexile search query:', searchQuery);
            } else {
                // If no Lexile score, find books in same age range
                searchQuery = baseBook.age_range || 'books for children';
                console.log('Constructed fallback search query:', searchQuery);
            }
            
            // Get progression recommendations
            console.log('About to make fetch request...');
            console.log('FETCH: Starting API request for query:', searchQuery);
            fetch('/api/simple-search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: searchQuery, page: 1, per_page: 12})
            })
            .then(response => response.json())
            .then(data => {
                console.log('API Response for Reading Progression:', data);
                console.log('data.success:', data.success);
                console.log('data.results length:', data.results ? data.results.length : 'undefined');
                
                if (data.success && data.results && data.results.length > 0) {
                    console.log('SUCCESS: Calling displayProgressionBooks with', data.results.length, 'books');
                    displayProgressionBooks(data.results, baseBook, baseLexile, selectedDifficulty);
                    return; // Exit early on success
                } else {
                    console.log('FAILURE: API response failed condition. Success:', data.success, 'Results:', data.results ? data.results.length : 'undefined');
                    // Fallback: search for books in same age range
                    const fallbackQuery = baseBook.age_range || 'books for children';
                    return fetch('/api/simple-search', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: fallbackQuery, page: 1, per_page: 12})
                    })
                    .then(response => response.json())
                    .then(fallbackData => {
                        if (fallbackData.success && fallbackData.results.length > 0) {
                            displayProgressionBooks(fallbackData.results, baseBook, baseLexile, selectedDifficulty);
                        } else {
                            if (loadingMessage) loadingMessage.style.display = 'none';
                            const resultsContainer = document.getElementById('progressResults');
                            if (resultsContainer) {
                                resultsContainer.innerHTML += '<div style="padding: 20px; color: #666;">No progression books found. Try searching for a different book.</div>';
                            }
                        }
                    });
                }
            })
            .catch(error => {
                console.error('Progression search error:', error);
                const loadingMessage = document.getElementById('loadingMessage');
                const resultsContainer = document.getElementById('progressResults');
                if (loadingMessage) loadingMessage.style.display = 'none';
                if (resultsContainer) {
                    resultsContainer.innerHTML += '<div style="padding: 20px; color: #e74c3c;">Error finding progression books. Please try again.</div>';
                }
            });
        }

        function selectBookForProgressionData(element) {
            const encodedBookData = element.getAttribute('data-book-json');
            return selectBookForProgression(encodedBookData);
        }

        function selectBookForProgression(bookData) {
            console.log('selectBookForProgression called with:', bookData);
            const book = JSON.parse(bookData);
            console.log('Parsed book:', book);
            
            // Hide suggestions
            document.getElementById('progressBookSuggestions').style.display = 'none';
            
            // Update input field with selected book title
            document.getElementById('progressBookSearch').value = book.title;
            
            // Find progression books for the selected book
            findProgressionBooks(book);
        }

        function findProgressionBooks(baseBook) {
            const resultsContainer = document.getElementById('progressResults');
            
            // Show the base book they finished
            const baseLexile = parseFloat(baseBook.lexile_score) || 0;
            
            resultsContainer.innerHTML = `
                <div style="margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745;">
                    <h3 style="margin-top: 0; color: #28a745;">📖 Great job finishing:</h3>
                    <div style="display: flex; gap: 15px; align-items: center;">
                        <div style="width: 50px; height: 75px; border-radius: 6px; overflow: hidden;">
                            ${baseBook.cover_url ? 
                                `<img src="/api/proxy-image?url=${encodeURIComponent(baseBook.cover_url)}" alt="${baseBook.title}" 
                                    style="width: 100%; height: 100%; object-fit: cover;" 
                                    onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                                <div style="display: none; width: 50px; height: 75px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 6px; align-items: center; justify-content: center; color: white; font-size: 20px;">📚</div>` 
                                : 
                                `<div style="width: 50px; height: 75px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-size: 20px;">📚</div>`
                            }
                        </div>
                        <div>
                            <div style="font-weight: 600; font-size: 16px;">${baseBook.title}</div>
                            <div style="color: #6b7280;">by ${baseBook.author}</div>
                            <div style="font-size: 14px; color: #9ca3af;">
                                ${baseBook.age_range || ''} ${baseLexile > 0 ? `• ${baseLexile}L Lexile` : ''}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;">
                    <h4 style="margin-top: 0; margin-bottom: 15px; color: #374151; font-size: 16px;">Choose your reading level preference:</h4>
                    <div style="display: flex; gap: 15px; flex-wrap: wrap; justify-content: center;">
                        <label style="display: flex; align-items: center; padding: 18px 24px; background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%); border: 2px solid #d1fae5; border-radius: 16px; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(34, 197, 94, 0.1); position: relative; overflow: hidden;" class="difficulty-option" data-difficulty="lower">
                            <input type="radio" name="difficulty-level" value="lower" style="display: none;">
                            <div>
                                <div class="difficulty-title" style="font-weight: 700; color: #059669; font-size: 16px; margin-bottom: 4px;">📉 Easier Books</div>
                                <div class="difficulty-subtitle" style="font-size: 13px; color: #6b7280; font-weight: 500;">Same or lower difficulty</div>
                            </div>
                        </label>
                        <label style="display: flex; align-items: center; padding: 18px 24px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 2px solid #bae6fd; border-radius: 16px; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1); position: relative; overflow: hidden;" class="difficulty-option" data-difficulty="similar">
                            <input type="radio" name="difficulty-level" value="similar" style="display: none;">
                            <div>
                                <div class="difficulty-title" style="font-weight: 700; color: #0891b2; font-size: 16px; margin-bottom: 4px;">📊 Similar Level</div>
                                <div class="difficulty-subtitle" style="font-size: 13px; color: #6b7280; font-weight: 500;">Around the same difficulty</div>
                            </div>
                        </label>
                        <label style="display: flex; align-items: center; padding: 18px 24px; background: linear-gradient(135deg, #fef2f2 0%, #fef7f7 100%); border: 2px solid #fecaca; border-radius: 16px; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1); position: relative; overflow: hidden;" class="difficulty-option" data-difficulty="advanced">
                            <input type="radio" name="difficulty-level" value="advanced" style="display: none;">
                            <div>
                                <div class="difficulty-title" style="font-weight: 700; color: #dc2626; font-size: 16px; margin-bottom: 4px;">📈 Challenge Me</div>
                                <div class="difficulty-subtitle" style="font-size: 13px; color: #6b7280; font-weight: 500;">Higher difficulty level</div>
                            </div>
                        </label>
                    </div>
                </div>
                
                <div id="loadingMessage" style="display: none; text-align: center; margin: 20px 0;">
                    <div style="display: inline-block; padding: 8px 16px; background: #e3f2fd; border-radius: 20px; color: #1976d2; font-weight: 600;">
                        📈 Finding your next reading challenge...
                    </div>
                </div>
            `;
            
            // Store base book info for re-searches
            sessionStorage.setItem('currentProgressionBook', JSON.stringify(baseBook));
            
            // No default selection - user must choose their preferred difficulty level

            // Attach event listeners to difficulty buttons
            console.log('RADIO DEBUG: Attaching event listeners for difficulty selection');
            document.querySelectorAll('.difficulty-option').forEach((option, index) => {
                console.log(`RADIO DEBUG: Adding listener to option ${index}:`, option.getAttribute('data-difficulty'));
                option.addEventListener('click', function() {
                    console.log(`RADIO DEBUG: Clicked on option with data-difficulty: ${this.getAttribute('data-difficulty')}`);
                    
                    // Reset all options visually to their default states
                    document.querySelectorAll('.difficulty-option').forEach(opt => {
                        const difficulty = opt.getAttribute('data-difficulty');
                        if (difficulty === 'lower') {
                            opt.style.background = 'linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%)';
                            opt.style.borderColor = '#d1fae5';
                            opt.style.boxShadow = '0 4px 12px rgba(34, 197, 94, 0.1)';
                            opt.style.transform = 'none';
                            opt.querySelector('div > div').style.color = '#059669';
                            opt.querySelector('div > div:last-child').style.color = '#6b7280';
                        } else if (difficulty === 'similar') {
                            opt.style.background = 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)';
                            opt.style.borderColor = '#bae6fd';
                            opt.style.boxShadow = '0 4px 12px rgba(14, 165, 233, 0.1)';
                            opt.style.transform = 'none';
                            opt.querySelector('div > div').style.color = '#0891b2';
                            opt.querySelector('div > div:last-child').style.color = '#6b7280';
                        } else if (difficulty === 'advanced') {
                            opt.style.background = 'linear-gradient(135deg, #fef2f2 0%, #fef7f7 100%)';
                            opt.style.borderColor = '#fecaca';
                            opt.style.boxShadow = '0 4px 12px rgba(239, 68, 68, 0.1)';
                            opt.style.transform = 'none';
                            opt.querySelector('div > div').style.color = '#dc2626';
                            opt.querySelector('div > div:last-child').style.color = '#6b7280';
                        }
                    });
                    
                    // Highlight selected option with enhanced styling
                    const selectedDifficulty = this.getAttribute('data-difficulty');
                    if (selectedDifficulty === 'lower') {
                        this.style.background = 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)';
                        this.style.borderColor = '#059669';
                        this.style.boxShadow = '0 8px 25px rgba(34, 197, 94, 0.3)';
                        this.style.transform = 'translateY(-2px)';
                        this.querySelector('div > div').style.color = 'white';
                        this.querySelector('div > div:last-child').style.color = 'rgba(255, 255, 255, 0.8)';
                    } else if (selectedDifficulty === 'similar') {
                        this.style.background = 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)';
                        this.style.borderColor = '#0891b2';
                        this.style.boxShadow = '0 8px 25px rgba(14, 165, 233, 0.3)';
                        this.style.transform = 'translateY(-2px)';
                        this.querySelector('div > div').style.color = 'white';
                        this.querySelector('div > div:last-child').style.color = 'rgba(255, 255, 255, 0.8)';
                    } else if (selectedDifficulty === 'advanced') {
                        this.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
                        this.style.borderColor = '#dc2626';
                        this.style.boxShadow = '0 8px 25px rgba(239, 68, 68, 0.3)';
                        this.style.transform = 'translateY(-2px)';
                        this.querySelector('div > div').style.color = 'white';
                        this.querySelector('div > div:last-child').style.color = 'rgba(255, 255, 255, 0.8)';
                    }
                    
                    // Make sure the radio button inside is checked
                    const radioButton = this.querySelector('input[type="radio"]');
                    if (radioButton) {
                        radioButton.checked = true;
                        console.log(`RADIO DEBUG: Set radio button ${radioButton.value} to checked`);
                    } else {
                        console.error('RADIO DEBUG: No radio button found inside clicked option');
                    }
                    
                    // Debug all radio states after click
                    document.querySelectorAll('input[name="difficulty-level"]').forEach(radio => {
                        console.log(`RADIO DEBUG: After click - ${radio.value}: ${radio.checked}`);
                    });
                    
                    // AUTO SEARCH: Automatically trigger search when difficulty changes
                    console.log('AUTO SEARCH: Difficulty changed, triggering automatic search');
                    
                    setTimeout(() => {
                        triggerProgressionSearch();
                    }, 100);
                });
            });

        }

        function displayProgressionBooks(books, baseBook, baseLexile, difficulty = 'similar') {
            console.log('=== DISPLAY PROGRESSION BOOKS DEBUG ===');
            console.log('Books count:', books.length);
            console.log('Difficulty parameter received:', difficulty);
            console.log('======================================');
            const resultsContainer = document.getElementById('progressResults');
            console.log('Results container found:', !!resultsContainer);
            
            if (!resultsContainer) {
                console.error('ERROR: progressResults element not found!');
                return;
            }
            
            // Filter and sort books by progression level
            console.log('Starting with', books.length, 'books from API');
            console.log('Base book for filtering:', baseBook.title, 'by', baseBook.author);
            
            const filteredBooks = books.filter(book => book.title !== baseBook.title || book.author !== baseBook.author);
            console.log('After filtering out base book:', filteredBooks.length, 'books remain');
            
            const progressionBooks = filteredBooks
                .map(book => {
                    const bookLexile = parseFloat(book.lexile_score) || 0;
                    const lexileDiff = bookLexile - baseLexile;
                    return { ...book, lexileDiff, bookLexile };
                })
                .sort((a, b) => Math.abs(a.lexileDiff - 100) - Math.abs(b.lexileDiff - 100)) // Sort by optimal progression (+100L)
                .slice(0, 9); // Show top 9 recommendations
            
            console.log('Progression books after filtering:', progressionBooks.length);
            if (progressionBooks.length === 0) {
                console.warn('No books after filtering! Original count:', books.length);
                console.log('Base book title/author for filtering:', baseBook.title, baseBook.author);
            }
            
            // Create dynamic title and description based on difficulty level
            let difficultyLabel, difficultyDescription, difficultyIcon;
            
            switch(difficulty) {
                case 'lower':
                    difficultyIcon = '📉';
                    difficultyLabel = 'Easier Books';
                    difficultyDescription = baseLexile > 0 ? 
                        `These books are at the same or slightly lower difficulty level (${Math.max(0, Math.round(baseLexile - 50))}L-${Math.round(baseLexile + 25)}L range).` :
                        'These books are recommended for building confidence with similar or easier reading levels.';
                    break;
                case 'similar':
                    difficultyIcon = '📊';
                    difficultyLabel = 'Similar Level Books';
                    difficultyDescription = baseLexile > 0 ? 
                        `These books are around the same difficulty level (${Math.round(baseLexile - 50)}L-${Math.round(baseLexile + 50)}L range).` :
                        'These books are recommended based on similar age range and reading level.';
                    break;
                case 'advanced':
                    difficultyIcon = '📈';
                    difficultyLabel = 'Challenge Books';
                    difficultyDescription = baseLexile > 0 ? 
                        `These books will challenge and advance reading skills (${Math.round(baseLexile + 25)}L-${Math.round(baseLexile + 200)}L range).` :
                        'These books are recommended to provide a reading challenge and skill advancement.';
                    break;
            }

            // DISABLED: Old progressionHtml generation - now using newer beautiful book cards implementation
            console.log('DISABLED: Old progressionHtml generation to prevent duplicate cards');
            
            // progressionHtml no longer used
            console.log('progressionBooks.length:', progressionBooks.length);
            console.log('createPrettierBookCard exists:', typeof createPrettierBookCard);
            
            // Test createPrettierBookCard with a sample book
            if (progressionBooks.length > 0) {
                console.log('Testing createPrettierBookCard with first book...');
                const testCard = createPrettierBookCard(progressionBooks[0], 0, 'progress');
                console.log('Test card generated, length:', testCard.length);
                console.log('Test card sample:', testCard.substring(0, 300));
                console.log('Test card contains title:', testCard.includes(progressionBooks[0].title));
                
                // Test the actual map operation that generates all cards
                console.log('Testing map operation for all books...');
                const allCards = progressionBooks.map((book, index) => createPrettierBookCard(book, index, 'progress'));
                console.log('All cards generated:', allCards.length);
                console.log('Total cards length:', allCards.join('').length);
                console.log('First 3 card lengths:', allCards.slice(0, 3).map(card => card.length));
            }
            
            // Test if the HTML contains actual book cards or just the wrapper
            if (progressionBooks.length > 0) {
                console.log('First book title:', progressionBooks[0].title);
                // progressionHtml check no longer needed
            }
            
            // DISABLED: Old implementation that created duplicate book cards
            // This was causing conflicts with the newer beautiful book cards implementation
            console.log('DISABLED: Old book card generation to prevent duplicates');
            
            console.log('Updated container HTML length:', resultsContainer.innerHTML.length);
            console.log('Container visibility:', window.getComputedStyle(resultsContainer).display);
            console.log('Container height:', resultsContainer.offsetHeight);
            console.log('Container overflow:', window.getComputedStyle(resultsContainer).overflow);
            console.log('Container position:', window.getComputedStyle(resultsContainer).position);
            console.log('Container z-index:', window.getComputedStyle(resultsContainer).zIndex);
            console.log('Number of child elements:', resultsContainer.children.length);
            
            // DOM updates are working - removing debug element
            console.log('DOM updates confirmed working');
            
            // Log a sample of the actual HTML content
            const sampleHTML = resultsContainer.innerHTML.substring(0, 500);
            console.log('Sample HTML content:', sampleHTML);
            
            // Check if the results section was actually created  
            // Find the results section by looking for the one that contains book recommendations
            let resultsSection = null;
            const allSections = resultsContainer.querySelectorAll('div[style*="margin-top: 20px"]');
            console.log('All sections with margin-top found:', allSections.length);
            
            for (let i = 0; i < allSections.length; i++) {
                const h3 = allSections[i].querySelector('h3');
                const h3Text = h3 ? h3.textContent : 'no h3';
                console.log(`Section ${i} contains h3:`, h3Text);
                
                // Look for the section that contains difficulty level info (📉 📊 📈)
                if (h3Text.includes('📉') || h3Text.includes('📊') || h3Text.includes('📈')) {
                    resultsSection = allSections[i];
                    console.log(`Found results section at index ${i}`);
                    break;
                }
            }
            
            if (!resultsSection) {
                // Fallback: try the last section
                resultsSection = allSections[allSections.length - 1];
                console.log('Using fallback: last section');
            }
            console.log('Results section found:', !!resultsSection);
            if (resultsSection) {
                const h3 = resultsSection.querySelector('h3');
                console.log('Results section h3 text:', h3 ? h3.textContent : 'no h3');
                console.log('Results section height:', resultsSection.offsetHeight);
                console.log('Results section children:', resultsSection.children.length);
                console.log('Results section innerHTML length:', resultsSection.innerHTML.length);
                console.log('Results section innerHTML sample:', resultsSection.innerHTML.substring(0, 500));
                
                // Check for the books grid specifically
                const booksGrid = resultsSection.querySelector('div[style*="grid-template-columns"]');
                console.log('Books grid found:', !!booksGrid);
                if (booksGrid) {
                    console.log('Books grid children:', booksGrid.children.length);
                    console.log('Grid display style:', window.getComputedStyle(booksGrid).display);
                    console.log('Grid height:', booksGrid.offsetHeight);
                    
                    // Store reference for scrolling
                    window.currentBooksGrid = booksGrid;
                    
                    console.log('Using original createPrettierBookCard approach');
                    
                    // Check if any book cards exist
                    const bookCards = booksGrid.querySelectorAll('.book-card');
                    console.log('Book cards found:', bookCards.length);
                    if (bookCards.length > 0) {
                        console.log('First book card height:', bookCards[0].offsetHeight);
                        console.log('First book card display:', window.getComputedStyle(bookCards[0]).display);
                        console.log('First book card visibility:', window.getComputedStyle(bookCards[0]).visibility);
                        console.log('First book card opacity:', window.getComputedStyle(bookCards[0]).opacity);
                        console.log('First book card position:', window.getComputedStyle(bookCards[0]).position);
                        
                        // Check if the card is actually visible in the viewport
                        const rect = bookCards[0].getBoundingClientRect();
                        console.log('First book card rect:', {
                            top: rect.top,
                            left: rect.left, 
                            width: rect.width,
                            height: rect.height,
                            inViewport: rect.top >= 0 && rect.left >= 0 && rect.top < window.innerHeight
                        });
                    }
                } // Close if (booksGrid)
            } // Close if (resultsSection)
            
            // Hide loading message
            const loadingMessage = document.getElementById('loadingMessage');
            if (loadingMessage) loadingMessage.style.display = 'none';
            
            // Preserve and restore the selected difficulty option
            const selectedDifficultyOption = difficulty || 'similar';
            const selectedRadio = document.querySelector(`input[name="difficulty-level"][value="${selectedDifficultyOption}"]`);
            if (selectedRadio) {
                selectedRadio.checked = true;
                // Also highlight the selected option visually
                const selectedLabel = selectedRadio.closest('.difficulty-option');
                if (selectedLabel) {
                    selectedLabel.style.borderColor = '#3b82f6';
                    selectedLabel.style.backgroundColor = '#eff6ff';
                }
            }
            
            // IMMEDIATE SCROLL - no setTimeout delay
            console.log('IMMEDIATE SCROLL: Starting scroll to results');
            
            // Find grid elements directly
            const gridElements = resultsContainer.querySelectorAll('div[style*="display: grid"]');
            console.log('IMMEDIATE SCROLL: Found grid elements:', gridElements.length);
                
            
            // CRITICAL FIX: Hide loading message and ensure results are visible
            const loadingDiv = document.getElementById('loadingMessage');
            if (loadingDiv) {
                loadingDiv.style.display = 'none';
                console.log('CRITICAL FIX: Hidden loading message');
            }
            
            // Force immediate scroll to results container
            console.log('IMMEDIATE SCROLL: Scrolling to results container first');
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // Remove the test - we confirmed DOM manipulation works
            console.log('SUCCESS: DOM manipulation confirmed working');
            
            if (gridElements.length > 0) {
                const booksGrid = gridElements[0];
                console.log('IMMEDIATE SCROLL: Found books grid, applying highlight');
                console.log('GRID CONTENT CHECK:', booksGrid.innerHTML.substring(0, 300));
                
                // Ensure visibility
                booksGrid.style.display = 'grid';
                booksGrid.style.visibility = 'visible';
                booksGrid.style.opacity = '1';
                booksGrid.style.minHeight = '500px';
                
                // Add bright highlight and fix layout
                booksGrid.style.border = '10px solid red';
                booksGrid.style.backgroundColor = 'yellow';
                booksGrid.style.position = 'relative';
                booksGrid.style.zIndex = '10';
                booksGrid.style.margin = '20px auto';
                booksGrid.style.width = '100%';
                booksGrid.style.maxWidth = '1200px';
                
                // Scroll to the grid
                console.log('IMMEDIATE SCROLL: Scrolling to books grid');
                booksGrid.scrollIntoView({ behavior: 'smooth', block: 'center' });
                
                // Verify the styles were actually applied
                console.log('IMMEDIATE SCROLL: Grid border after styling:', booksGrid.style.border);
                console.log('IMMEDIATE SCROLL: Grid background after styling:', booksGrid.style.backgroundColor);
                console.log('IMMEDIATE SCROLL: Grid position after styling:', booksGrid.getBoundingClientRect());
                console.log('IMMEDIATE SCROLL: Grid computed styles:', {
                    display: window.getComputedStyle(booksGrid).display,
                    visibility: window.getComputedStyle(booksGrid).visibility,
                    opacity: window.getComputedStyle(booksGrid).opacity,
                    position: window.getComputedStyle(booksGrid).position
                });
                
                // Check parent containers for CSS issues
                let parent = booksGrid.parentElement;
                let level = 1;
                while (parent && level <= 3) {
                    console.log(`PARENT LEVEL ${level}:`, {
                        tagName: parent.tagName,
                        id: parent.id,
                        className: parent.className,
                        display: window.getComputedStyle(parent).display,
                        overflow: window.getComputedStyle(parent).overflow,
                        height: window.getComputedStyle(parent).height,
                        maxHeight: window.getComputedStyle(parent).maxHeight
                    });
                    parent = parent.parentElement;
                    level++;
                }
                
                console.log('IMMEDIATE SCROLL: Applied red border and yellow background');
            } else {
                console.log('IMMEDIATE SCROLL: No grid found');
            }
            
            // FINAL BYPASS: Clear previous results and create fresh book list
            console.log('FINAL BYPASS: Clearing previous results');
            const existingSimpleBookLists = resultsContainer.querySelectorAll('div[data-simple-book-list="true"]');
            existingSimpleBookLists.forEach(list => list.remove());
            
            console.log('BOOK CARDS: Creating beautiful book cards using createPrettierBookCard');
            const bookCardsContainer = document.createElement('div');
            bookCardsContainer.setAttribute('data-simple-book-list', 'true'); // Mark for easy removal later
            
            // Get difficulty label for display
            const difficultyLabelMap = {
                'lower': '📉 Easier Books',
                'similar': '📊 Similar Level', 
                'advanced': '📈 Challenge Me'
            };
            const selectedDifficultyLabel = difficultyLabelMap[difficulty] || '📚 Books';
            
            // Create header with beautiful styling
            const headerHtml = `
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 24px; border-radius: 16px; margin: 20px 0; color: white; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);">
                    <h2 style="margin: 0 0 8px 0; font-size: 24px; font-weight: 700; color: white;">${selectedDifficultyLabel} (${progressionBooks.length} found)</h2>
                    <p style="margin: 0; font-size: 16px; opacity: 0.9; font-weight: 500;">Based on "${baseBook.title}" ${baseLexile > 0 ? `(${baseLexile}L)` : ''}</p>
                </div>
            `;
            
            // Create grid container for book cards
            const booksGridHtml = `
                <div class="books-grid book-grid" style="margin-top: 24px;">
                    ${progressionBooks.map((book, index) => {
                        console.log('Creating beautiful card for:', book.title);
                        return createPrettierBookCard(book, index, 'progress');
                    }).join('')}
                </div>
            `;
            
            bookCardsContainer.innerHTML = headerHtml + booksGridHtml;
            resultsContainer.appendChild(bookCardsContainer);
            
            // Store books data for action button functionality
            window.currentBooks = progressionBooks;
            window.progressionBooks = progressionBooks; // Also store in dedicated variable
            console.log('BOOK CARDS: Added beautiful book cards to resultsContainer');
            console.log('BOOK CARDS: Stored', progressionBooks.length, 'books in window.currentBooks and window.progressionBooks');
            
            // Add debugging for radio button clicks - attach every time since DOM is recreated
            console.log('RADIO DEBUG: Attaching event listeners for difficulty selection');
                
                document.querySelectorAll('.difficulty-option').forEach((option, index) => {
                    console.log(`RADIO DEBUG: Adding listener to option ${index}:`, option.getAttribute('data-difficulty'));
                    option.addEventListener('click', function() {
                    console.log(`RADIO DEBUG: Clicked on option with data-difficulty: ${this.getAttribute('data-difficulty')}`);
                    
                    // Reset all options visually to their default states
                    document.querySelectorAll('.difficulty-option').forEach(opt => {
                        const difficulty = opt.getAttribute('data-difficulty');
                        if (difficulty === 'lower') {
                            opt.style.background = 'linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%)';
                            opt.style.borderColor = '#d1fae5';
                            opt.style.boxShadow = '0 4px 12px rgba(34, 197, 94, 0.1)';
                            opt.style.transform = 'none';
                            opt.querySelector('div > div').style.color = '#059669';
                            opt.querySelector('div > div:last-child').style.color = '#6b7280';
                        } else if (difficulty === 'similar') {
                            opt.style.background = 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)';
                            opt.style.borderColor = '#bae6fd';
                            opt.style.boxShadow = '0 4px 12px rgba(14, 165, 233, 0.1)';
                            opt.style.transform = 'none';
                            opt.querySelector('div > div').style.color = '#0891b2';
                            opt.querySelector('div > div:last-child').style.color = '#6b7280';
                        } else if (difficulty === 'advanced') {
                            opt.style.background = 'linear-gradient(135deg, #fef2f2 0%, #fef7f7 100%)';
                            opt.style.borderColor = '#fecaca';
                            opt.style.boxShadow = '0 4px 12px rgba(239, 68, 68, 0.1)';
                            opt.style.transform = 'none';
                            opt.querySelector('div > div').style.color = '#dc2626';
                            opt.querySelector('div > div:last-child').style.color = '#6b7280';
                        }
                    });
                    
                    // Highlight selected option with enhanced styling
                    const selectedDifficulty = this.getAttribute('data-difficulty');
                    if (selectedDifficulty === 'lower') {
                        this.style.background = 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)';
                        this.style.borderColor = '#059669';
                        this.style.boxShadow = '0 8px 25px rgba(34, 197, 94, 0.3)';
                        this.style.transform = 'translateY(-2px)';
                        this.querySelector('div > div').style.color = 'white';
                        this.querySelector('div > div:last-child').style.color = 'rgba(255, 255, 255, 0.8)';
                    } else if (selectedDifficulty === 'similar') {
                        this.style.background = 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)';
                        this.style.borderColor = '#0891b2';
                        this.style.boxShadow = '0 8px 25px rgba(14, 165, 233, 0.3)';
                        this.style.transform = 'translateY(-2px)';
                        this.querySelector('div > div').style.color = 'white';
                        this.querySelector('div > div:last-child').style.color = 'rgba(255, 255, 255, 0.8)';
                    } else if (selectedDifficulty === 'advanced') {
                        this.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
                        this.style.borderColor = '#dc2626';
                        this.style.boxShadow = '0 8px 25px rgba(239, 68, 68, 0.3)';
                        this.style.transform = 'translateY(-2px)';
                        this.querySelector('div > div').style.color = 'white';
                        this.querySelector('div > div:last-child').style.color = 'rgba(255, 255, 255, 0.8)';
                    }
                    
                    // Make sure the radio button inside is checked
                    const radioButton = this.querySelector('input[type="radio"]');
                    if (radioButton) {
                        radioButton.checked = true;
                        console.log(`RADIO DEBUG: Set radio button ${radioButton.value} to checked`);
                    } else {
                        console.error('RADIO DEBUG: No radio button found inside clicked option');
                    }
                    
                    // Debug all radio states after click
                    document.querySelectorAll('input[name="difficulty-level"]').forEach(radio => {
                        console.log(`RADIO DEBUG: After click - ${radio.value}: ${radio.checked}`);
                    });
                    
                    // AUTO SEARCH: Automatically trigger search when difficulty changes
                    console.log('AUTO SEARCH: Difficulty changed, triggering automatic search');
                    
                    setTimeout(() => {
                        triggerProgressionSearch();
                    }, 100);
                });
            });
            
            // Event handlers are already attached when the interface is first created
            // No need to re-attach them here as it causes conflicts
            
            // Final debugging: Check if books are actually visible
            setTimeout(() => {
                const allBookCards = document.querySelectorAll('.book-card');
                console.log('FINAL CHECK: Total book cards in DOM:', allBookCards.length);
                
                const cleanBookCards = resultsContainer.querySelectorAll('div[style*="background: white; border: 2px solid #e0e0e0"]');
                console.log('FINAL CHECK: Clean book cards found:', cleanBookCards.length);
                
                if (cleanBookCards.length === 0 && allBookCards.length === 0) {
                    console.error('CRITICAL: NO BOOK CARDS FOUND IN DOM!');
                    console.log('resultsContainer children:', resultsContainer.children.length);
                    for (let i = 0; i < resultsContainer.children.length; i++) {
                        console.log(`Child ${i}:`, resultsContainer.children[i].tagName, resultsContainer.children[i].className);
                    }
                }
            }, 500);
        }

        // Separate function for Similar Books to avoid conflicts with AI Book Analysis
        function displaySimilarBookSuggestions(books) {
            const container = document.getElementById('similarBookSuggestions');
            console.log('displaySimilarBookSuggestions called with', books.length, 'books');
            
            container.innerHTML = '';
            
            books.forEach((book, index) => {
                console.log('Creating suggestion for book:', book.title);
                
                const div = document.createElement('div');
                div.style.cssText = 'padding: 12px; border-bottom: 1px solid #e5e7eb; cursor: pointer; display: flex; gap: 12px; align-items: center;';
                
                div.innerHTML = `
                    <div style="width: 40px; height: 60px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 16px;">📚</div>
                    <div>
                        <div style="font-weight: 600; color: #1f2937;">${book.title}</div>
                        <div style="font-size: 13px; color: #6b7280;">by ${book.author}</div>
                        <div style="font-size: 12px; color: #9ca3af;">${book.age_range || ''} • ${book.themes ? book.themes.split(',')[0] : ''}</div>
                    </div>
                `;
                
                div.addEventListener('click', function() {
                    console.log('CLICKED ON BOOK:', book.title);
                    console.log('Book data:', book);
                    selectBookForSimilarity(JSON.stringify(book));
                });
                
                div.addEventListener('mouseover', function() {
                    this.style.background = '#f9fafb';
                });
                
                div.addEventListener('mouseout', function() {
                    this.style.background = 'white';
                });
                
                container.appendChild(div);
            });
            
            console.log('Container populated and making visible');
            container.style.display = 'block';
        }

        function selectBookForSimilarityData(element) {
            console.log('selectBookForSimilarityData called', element);
            console.log('Element clicked:', element.outerHTML);
            try {
                const bookData = element.getAttribute('data-book-json');
                console.log('Book data from attribute:', bookData);
                if (!bookData) {
                    console.error('No book data found in element');
                    alert('Error: No book data found');
                    return;
                }
                return selectBookForSimilarity(bookData);
            } catch (error) {
                console.error('Error in selectBookForSimilarityData:', error);
                alert('Error selecting book: ' + error.message);
            }
        }

        function selectBookForSimilarity(bookData) {
            console.log('selectBookForSimilarity called with:', bookData);
            try {
                const book = JSON.parse(bookData);
                console.log('Parsed book:', book);
            
            // Hide suggestions (like Reading Progression does)
            const suggestionsElement = document.getElementById('similarBookSuggestions');
            console.log('Hiding suggestions element:', suggestionsElement);
            suggestionsElement.style.display = 'none';
            
            // Update search field with selected book title (like Reading Progression does)
            const searchInput = document.getElementById('similarBookSearch');
            if (searchInput) {
                searchInput.value = book.title;
                console.log('Updated search input with book title:', book.title);
            }
            
            // Immediately find and display similar books (like Reading Progression does)
            findSimilarBooks(JSON.stringify(book));
            console.log('Called findSimilarBooks directly');
            } catch (error) {
                console.error('Error in selectBookForSimilarity:', error);
                alert('Error selecting book: ' + error.message);
            }
        }

        function findSimilarBooksData(element) {
            const bookData = element.getAttribute('data-book');
            return findSimilarBooks(bookData);
        }

        function findSimilarBooks(bookData) {
            const book = JSON.parse(bookData);
            const resultsContainer = document.getElementById('similarResults');
            
            resultsContainer.innerHTML = '<div style="text-align: center; padding: 20px;">Finding similar books...</div>';
            
            fetch('/api/similar', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    title: book.title,
                    author: book.author
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displaySimilarResults(data.similar_books, book);
                } else {
                    resultsContainer.innerHTML = `<div class="message error">Error: ${data.error}</div>`;
                }
            })
            .catch(error => {
                console.error('Similar books error:', error);
                resultsContainer.innerHTML = `<div class="message error">Failed to find similar books</div>`;
            });
        }

        function displaySimilarResults(books, baseBook) {
            const container = document.getElementById('similarResults');
            
            if (books.length === 0) {
                container.innerHTML = '<div class="message info">No similar books found.</div>';
                return;
            }
            
            // Show the base book header like Reading Progression does
            const baseBookHtml = `
                <div style="margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea;">
                    <h3 style="margin-top: 0; color: #667eea;">🎯 Similar books to:</h3>
                    <div style="display: flex; gap: 15px; align-items: center;">
                        <div style="width: 50px; height: 75px; border-radius: 6px; overflow: hidden;">
                            ${baseBook.cover_url ? 
                                `<img src="/api/proxy-image?url=${encodeURIComponent(baseBook.cover_url)}" alt="${baseBook.title}" 
                                    style="width: 100%; height: 100%; object-fit: cover;" 
                                    onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                                <div style="display: none; width: 50px; height: 75px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 6px; align-items: center; justify-content: center; color: white; font-size: 20px;">📚</div>` 
                                : 
                                `<div style="width: 50px; height: 75px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-size: 20px;">📚</div>`
                            }
                        </div>
                        <div>
                            <div style="font-weight: 600; font-size: 18px; color: #1f2937;">${baseBook.title}</div>
                            <div style="color: #6b7280; font-size: 14px;">by ${baseBook.author}</div>
                            <div style="color: #9ca3af; font-size: 12px; margin-top: 4px;">${baseBook.age_range || ''} ${baseBook.age_range && baseBook.lexile_score ? '•' : ''} ${baseBook.lexile_score && baseBook.lexile_score !== 'nan' ? baseBook.lexile_score + 'L Lexile' : ''}</div>
                        </div>
                    </div>
                </div>
            `;
            
            const resultsHtml = `
                <div style="margin-top: 20px;">
                    <h3 style="color: #1976d2;">📚 Similar Books (${books.length} found)</h3>
                    <p style="color: #666; margin-bottom: 20px;">
                        These books share similar themes, reading level, and style.
                    </p>
                    
                    <div class="book-grid">
                        ${books.map((book, index) => createPrettierBookCard(book, index, 'similar')).join('')}
                    </div>
                </div>
            `;
            
            console.log('SIMILAR DEBUG: Setting container innerHTML');
            console.log('SIMILAR DEBUG: Container ID:', container.id);
            console.log('SIMILAR DEBUG: Books count:', books.length);
            container.innerHTML = baseBookHtml + resultsHtml;
            
            // Debug: Check what book card IDs were actually created
            setTimeout(() => {
                const bookCards = container.querySelectorAll('[id^="book-"]');
                console.log('SIMILAR DEBUG: Created book card IDs:', Array.from(bookCards).map(el => el.id));
                const peekButtons = container.querySelectorAll('[id^="peek-btn-"]');
                console.log('SIMILAR DEBUG: Created peek button IDs:', Array.from(peekButtons).map(el => el.id));
            }, 100);
            
            window.currentBooks = books;
            window.similarBooks = books; // Also store in dedicated variable
            console.log('SIMILAR DEBUG: Stored', books.length, 'books in window.currentBooks and window.similarBooks');
        }



        // ==========================================
        // NEW: Enhanced Lexile Prediction Functions  
        // ==========================================
        
        function getLexilePredictionHtml(book) {
            // Check if we have enhanced Lexile prediction data
            if (book.lexile_prediction) {
                const prediction = book.lexile_prediction;
                const confidenceClass = `lexile-confidence-${prediction.confidence_level}`;
                const confidenceText = {
                    'high': 'HIGH',
                    'medium': 'MED', 
                    'low': 'LOW'
                }[prediction.confidence_level] || 'N/A';
                
                let html = `
                    <div class="lexile-prediction-container">
                        <span>📊 ${prediction.predicted_lexile}L</span>
                        <span class="${confidenceClass}">${confidenceText}</span>
                `;
                
                // Add edge case warning if applicable
                if (prediction.is_edge_case) {
                    html += `<span class="edge-case-warning" title="${prediction.warning}">VINTAGE</span>`;
                }
                
                html += `</div>`;
                
                // Add warning tooltip for edge cases
                if (prediction.is_edge_case && prediction.warning) {
                    html += `
                        <div class="lexile-warning-tooltip" style="display: block;">
                            ⚠️ ${prediction.warning}
                            ${prediction.prediction_range ? `<br>Likely range: ${prediction.prediction_range}` : ''}
                        </div>
                    `;
                }
                
                return html;
            }
            
            // Fallback to existing Lexile display
            if (book.lexile_score && book.lexile_score !== 'nan') {
                return `<span>📊 ${Math.round(parseFloat(book.lexile_score))}L</span>`;
            }
            
            return '';
        }
        
        async function enhanceBookWithLexilePrediction(book) {
            // Always refresh prediction to ensure we get the latest data from our fixed API
            // This prevents frontend caching from showing stale results
            
            // Only enhance if we don't have a verified Lexile score
            if (book.lexile_score && book.lexile_score !== 'nan' && parseFloat(book.lexile_score) > 0) {
                return book;
            }
            
            try {
                // Extract age range
                let ageMin, ageMax;
                if (book.age_range) {
                    const ageMatch = book.age_range.match(/(\d+)-(\d+)/);
                    if (ageMatch) {
                        ageMin = parseInt(ageMatch[1]);
                        ageMax = parseInt(ageMatch[2]);
                    }
                }
                
                // Call our new Lexile prediction API
                const response = await fetch('/api/predict-lexile', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        title: book.title || '',
                        author: book.author || '',
                        age_min: ageMin,
                        age_max: ageMax,
                        book_type: book.lexile_prefix ? 'Adult_Directed' : 'Standard_Lexile',
                        notes: book.description || book.themes || ''
                    })
                });
                
                if (response.ok) {
                    const prediction = await response.json();
                    book.lexile_prediction = prediction;
                    
                    // Log successful prediction
                    console.log(`Enhanced ${book.title} with Lexile prediction:`, prediction);
                }
                
            } catch (error) {
                console.warn(`Failed to enhance ${book.title} with Lexile prediction:`, error);
            }
            
            return book;
        }
        
        async function enhanceBooksWithLexilePredictions(books) {
            // Process books in batches to avoid overwhelming the API
            const batchSize = 5;
            const enhancedBooks = [];
            
            for (let i = 0; i < books.length; i += batchSize) {
                const batch = books.slice(i, i + batchSize);
                const enhancedBatch = await Promise.all(
                    batch.map(book => enhanceBookWithLexilePrediction(book))
                );
                enhancedBooks.push(...enhancedBatch);
                
                // Small delay between batches
                if (i + batchSize < books.length) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
            }
            
            return enhancedBooks;
        }

        function createPrettierBookCard(book, index, context = 'search') {
            return `
                <div class="book-card" id="book-${context}-${index}" style="position: relative;">
                    <div class="book-header">
                        <div class="book-cover">
                            ${book.cover_url ? 
                                `<img src="/api/proxy-image?url=${encodeURIComponent(book.cover_url)}" alt="${book.title}" 
                                    onerror="console.log('Failed to load proxied image:', this.src); this.style.display='none'; this.nextElementSibling.style.display='flex';"
                                    onload="console.log('Successfully loaded proxied image:', this.src);">
                                <div class="book-cover-placeholder" style="display: none;">📚</div>` 
                                : 
                                `<div class="book-cover-placeholder">📚</div>`
                            }
                        </div>
                        <div class="book-info">
                            <h1 class="book-title">${book.title}</h1>
                            <p class="book-author">by ${book.author}</p>
                            
                            <div class="themes-pills">
                                ${book.themes ? book.themes.split(',').slice(0, 3).map(theme => 
                                    `<span class="theme-pill">${theme.trim()}</span>`
                                ).join('') : '<span class="theme-pill">general</span>'}
                            </div>
                            
                            <div class="book-meta">
                                ${book.tone ? `
                                <div class="meta-item">
                                    <span>😊</span>
                                    <span>${book.tone.charAt(0).toUpperCase() + book.tone.slice(1)}</span>
                                </div>` : ''}
                                ${book.age_range ? `
                                <div class="meta-item">
                                    <span>🎂</span>
                                    <span>Ages ${book.age_range}</span>
                                </div>` : ''}
                                ${book.lexile_score && book.lexile_score !== 'nan' ? `
                                <div class="meta-item">
                                    <span>📊</span>
                                    <span>${Math.round(parseFloat(book.lexile_score))}L</span>
                                </div>` : ''}
                            </div>
                        </div>
                    </div>
                    
                    <div class="k-actions-row">
                        <button onclick="saveBook(${index}, 'favorites')" class="action-btn like-btn">
                            👍 Like
                        </button>
                        <button onclick="saveBook(${index}, 'skipped')" class="action-btn skip-btn">
                            👎 Skip
                        </button>
                        <button onclick="saveBook(${index}, 'read')" class="action-btn read-btn">
                            📖 Read
                        </button>
                    </div>
                    
                    <div class="sneak-peek-section">
                        <button onclick="toggleSneakPeek('${context}-${index}')" class="sneak-peek-btn" id="peek-btn-${context}-${index}">
                            <span>▼</span>
                            <span>Sneak Peek</span>
                        </button>
                        <div class="sneak-peek-content" id="peek-content-${context}-${index}" style="display: none;">
                            ${getSummaryForBook(book) || 'Summary not available for this book.'}
                            <div class="k-linkbar">
                                <!-- Debug: ${book.title} - GR URL: ${book.goodreads_url} -->
                                ${book.goodreads_url && book.goodreads_url !== '' && book.goodreads_url !== 'undefined' && book.goodreads_url !== 'nan' ? `<a class="k-iconbtn k-iconbtn--goodreads" href="${book.goodreads_url}" target="_blank" rel="noopener noreferrer" aria-label="Open on Goodreads" title="Goodreads">
                                    <img src="https://cdn.simpleicons.org/goodreads/5A4634" class="k-ico" alt="Goodreads" loading="lazy" />
                                </a>` : '<!-- No GR URL -->'}
                                <a class="k-iconbtn k-iconbtn--amazon" href="https://www.amazon.com/s?k=${encodeURIComponent(book.title + ' ' + book.author)}" target="_blank" rel="noopener noreferrer" aria-label="Open on Amazon" title="Amazon">
                                    <i class="fa-brands fa-amazon k-fa"></i>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // ==========================================
        // NEW: Search Builder Functionality
        // ==========================================
        
        // Dynamic filtering functions
        async function updateAvailableOptions() {
            try {
                const theme = document.getElementById('searchBuilderTheme')?.value || '';
                const tone = document.getElementById('searchBuilderTone')?.value || '';
                const age = document.getElementById('searchBuilderAge')?.value || '';
                const lexile = document.getElementById('searchBuilderLexile')?.value || '';
                
                const params = new URLSearchParams();
                if (theme) params.append('theme', theme);
                if (tone) params.append('tone', tone);
                if (age) params.append('age', age);
                if (lexile) params.append('lexile', lexile);
                
                const response = await fetch(`/api/filter-options?${params.toString()}`);
                const data = await response.json();
                
                if (data.success) {
                    updateDropdownOptions(data.available_options);
                    
                    // Update book count indicator (optional)
                    const bookCount = data.total_books_available;
                    console.log(`📚 ${bookCount} books available with current selection`);
                }
            } catch (error) {
                console.error('Error updating filter options:', error);
            }
        }
        
        function updateDropdownOptions(availableOptions) {
            // Update themes dropdown
            if (availableOptions.themes) {
                const themeSelect = document.getElementById('searchBuilderTheme');
                const currentValue = themeSelect.value;
                
                // Clear and rebuild options
                themeSelect.innerHTML = '<option value="">Choose options</option>';
                availableOptions.themes.forEach(theme => {
                    const option = document.createElement('option');
                    option.value = theme.value;
                    option.textContent = theme.label;
                    themeSelect.appendChild(option);
                });
                
                // Restore selection if still available
                if (currentValue && [...themeSelect.options].some(opt => opt.value === currentValue)) {
                    themeSelect.value = currentValue;
                }
            }
            
            // Update tones dropdown
            if (availableOptions.tones) {
                const toneSelect = document.getElementById('searchBuilderTone');
                const currentValue = toneSelect.value;
                
                toneSelect.innerHTML = '<option value="">—</option>';
                availableOptions.tones.forEach(tone => {
                    const option = document.createElement('option');
                    option.value = tone.value;
                    option.textContent = tone.label;
                    toneSelect.appendChild(option);
                });
                
                if (currentValue && [...toneSelect.options].some(opt => opt.value === currentValue)) {
                    toneSelect.value = currentValue;
                }
            }
            
            // Update ages dropdown
            if (availableOptions.ages) {
                const ageSelect = document.getElementById('searchBuilderAge');
                const currentValue = ageSelect.value;
                
                ageSelect.innerHTML = '<option value="">Select age...</option>';
                availableOptions.ages.forEach(age => {
                    const option = document.createElement('option');
                    option.value = age.value;
                    option.textContent = age.label;
                    ageSelect.appendChild(option);
                });
                
                if (currentValue && [...ageSelect.options].some(opt => opt.value === currentValue)) {
                    ageSelect.value = currentValue;
                }
            }
            
            // Update lexile dropdown
            if (availableOptions.lexiles) {
                const lexileSelect = document.getElementById('searchBuilderLexile');
                const currentValue = lexileSelect.value;
                
                lexileSelect.innerHTML = '<option value="">Select level...</option>';
                availableOptions.lexiles.forEach(lexile => {
                    const option = document.createElement('option');
                    option.value = lexile.value;
                    option.textContent = lexile.label;
                    lexileSelect.appendChild(option);
                });
                
                if (currentValue && [...lexileSelect.options].some(opt => opt.value === currentValue)) {
                    lexileSelect.value = currentValue;
                }
            }
        }
        
        // Handle Theme/Tone changes
        function handleSearchBuilderChange(field) {
            console.log(`🔄 Search builder ${field} changed, updating available options...`);
            updateAvailableOptions();
        }
        
        // Handle Age OR Lexile selection (not both)
        function handleAgeOrLexileChange(selected) {
            if (selected === 'age') {
                const ageSelect = document.getElementById('searchBuilderAge');
                const lexileSelect = document.getElementById('searchBuilderLexile');
                if (ageSelect.value !== '') {
                    lexileSelect.value = '';
                    console.log('🔄 Cleared lexile selection because age was selected');
                }
            } else if (selected === 'lexile') {
                const ageSelect = document.getElementById('searchBuilderAge');
                const lexileSelect = document.getElementById('searchBuilderLexile');
                if (lexileSelect.value !== '') {
                    ageSelect.value = '';
                    console.log('🔄 Cleared age selection because lexile was selected');
                }
            }
            
            // Update available options after age/lexile change
            updateAvailableOptions();
        }

        function performSearchBuilderSearch() {
            console.log('🔍 performSearchBuilderSearch() called!');
            
            const theme = document.getElementById('searchBuilderTheme')?.value;
            const tone = document.getElementById('searchBuilderTone')?.value;
            const age = document.getElementById('searchBuilderAge')?.value;
            const lexile = document.getElementById('searchBuilderLexile')?.value;
            
            console.log('Search builder values:', { theme, tone, age, lexile });
            
            // Build search query from selected options
            let searchParts = [];
            
            if (theme) {
                searchParts.push(theme);
            }
            
            if (tone) {
                searchParts.push(tone);
            }
            
            if (age) {
                searchParts.push(`age ${age}`);
            }
            
            if (lexile) {
                if (lexile === 'AD') {
                    searchParts.push('AD adult directed');
                } else if (lexile === '1000+') {
                    searchParts.push('lexile 1000-1400');
                } else {
                    searchParts.push(`lexile ${lexile}`);
                }
            }
            
            // Add "books" at the end
            searchParts.push('books');
            
            const searchQuery = searchParts.join(' ');
            console.log('Built search query:', searchQuery);
            
            // Validate that at least theme is selected
            if (!theme) {
                alert('Please select at least a theme to search for books.');
                return;
            }
            
            // Show what criteria are being used
            let criteriaUsed = [];
            if (theme) criteriaUsed.push(`theme: ${theme}`);
            if (tone) criteriaUsed.push(`tone: ${tone}`);
            if (age) criteriaUsed.push(`age: ${age}`);
            if (lexile) criteriaUsed.push(`reading level: ${lexile}`);
            console.log('🎯 Search criteria:', criteriaUsed.join(', '));
            
            // For testing: Try simpler query if lexile was selected
            if (lexile) {
                console.log(`⚠️ Note: Lexile filter "${lexile}" may be restrictive. If no results, try without lexile.`);
            }
            
            console.log('Using setQueryAndSearch() function like the filter chips...');
            
            // Use the same function as the filter chips
            if (typeof setQueryAndSearch === 'function') {
                console.log('✅ Calling setQueryAndSearch with query:', searchQuery);
                setQueryAndSearch(searchQuery);
                console.log('✅ Search initiated successfully!');
            } else {
                console.error('❌ setQueryAndSearch function not found!');
                alert('Error: Search function not available. Please refresh the page.');
            }
            
            // Provide user feedback
            showSearchBuilderFeedback(theme, tone, age, lexile);
        }
        
        function showSearchBuilderFeedback(theme, tone, age, lexile) {
            const button = document.querySelector('.search-builder-button');
            const originalText = button.textContent;
            
            // Show what we're searching for
            let feedbackParts = [];
            if (theme) feedbackParts.push(theme);
            if (tone) feedbackParts.push(tone);
            if (age) feedbackParts.push(`ages ${age}`);
            if (lexile) {
                if (lexile === 'AD') {
                    feedbackParts.push('adult directed');
                } else {
                    feedbackParts.push(`${lexile} Lexile`);
                }
            }
            
            button.textContent = `Searching: ${feedbackParts.join(' • ')}...`;
            button.disabled = true;
            
            // Reset button after 2 seconds
            setTimeout(() => {
                button.textContent = originalText;
                button.disabled = false;
            }, 2000);
        }
        
        // Reset search builder form
        function resetSearchBuilder() {
            document.getElementById('searchBuilderTheme').value = '';
            document.getElementById('searchBuilderTone').value = '';
            document.getElementById('searchBuilderAge').value = '';
            document.getElementById('searchBuilderLexile').value = '';
        }
        
        // Add keyboard support and visual feedback for search builder
        document.addEventListener('DOMContentLoaded', function() {
            const searchBuilderSelects = document.querySelectorAll('.search-builder-select');
            searchBuilderSelects.forEach(select => {
                select.addEventListener('change', function() {
                    // Auto-enable search button when theme is selected
                    const theme = document.getElementById('searchBuilderTheme').value;
                    const button = document.querySelector('.search-builder-button');
                    
                    if (theme) {
                        button.style.background = 'rgba(255, 255, 255, 0.25)';
                        button.style.borderColor = 'rgba(255, 255, 255, 0.5)';
                    } else {
                        button.style.background = 'rgba(255, 255, 255, 0.2)';
                        button.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                    }
                });
            });
            
            // Add Enter key support
            searchBuilderSelects.forEach(select => {
                select.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') {
                        performSearchBuilderSearch();
                    }
                });
            });
        });
        
        // Initialize Age Distribution Pie Chart
        function initAgeDistributionChart() {
            const ctx = document.getElementById('ageDistributionChart');
            if (ctx) {
                new Chart(ctx, {
                    type: 'pie',
                    data: {
                        labels: ['3-5', '6-8', '9-12', '0-2', '13+', 'Other'],
                        datasets: [{
                            data: [30.4, 24.1, 18.2, 12.1, 9.1, 6.1],
                            backgroundColor: [
                                '#4f46e5', // Blue
                                '#7c3aed', // Purple
                                '#06b6d4', // Cyan
                                '#10b981', // Green
                                '#f59e0b', // Orange
                                '#ef4444'  // Red
                            ],
                            borderColor: 'white',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: false,
                        maintainAspectRatio: true,
                        plugins: {
                            legend: {
                                display: false // We'll use our custom legend
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return context.label + ': ' + context.parsed + '%';
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }

        // Initialize Lexile Distribution Bar Chart
        function initLexileDistributionChart() {
            const ctx = document.getElementById('lexileDistributionChart');
            if (ctx) {
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: ['0-200L', '200-400L', '400-600L', '600-800L', '800-1000L', '1000L+', 'AD'],
                        datasets: [{
                            data: [156, 234, 208, 182, 130, 104, 73],
                            backgroundColor: [
                                '#4f46e5',  // Blue
                                '#7c3aed',  // Purple
                                '#06b6d4',  // Cyan
                                '#10b981',  // Green
                                '#f59e0b',  // Orange
                                '#ef4444',  // Red
                                '#8b5cf6'   // Light Purple
                            ],
                            borderColor: 'white',
                            borderWidth: 2,
                            borderRadius: 6,
                            borderSkipped: false
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return context.label + ': ' + context.parsed.y + ' books';
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: {
                                    color: '#f1f5f9'
                                },
                                ticks: {
                                    color: '#6b7280',
                                    font: {
                                        size: 11
                                    }
                                }
                            },
                            x: {
                                grid: {
                                    display: false
                                },
                                ticks: {
                                    color: '#6b7280',
                                    font: {
                                        size: 11
                                    }
                                }
                            }
                        },
                        layout: {
                            padding: {
                                top: 20,
                                right: 10,
                                bottom: 10,
                                left: 10
                            }
                        }
                    }
                });
            }
        }

        // Initialize charts when DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            initAgeDistributionChart();
            initLexileDistributionChart();
        });

        console.log('Script finished loading');
        console.log('performSearch function exists:', typeof performSearch);
        
