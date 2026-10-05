
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
import hashlib
import re

@dataclass
class Ticket:
    """Represents a support ticket or issue."""
    id: str
    customer_id: str
    order_id: Optional[str] = None
    subject: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "open"  # open, resolved, closed
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # For semantic similarity
    
    def compute_hash(self) -> str:
        """Compute a content hash for duplicate detection."""
        content = f"{self.customer_id}:{self.order_id}:{self.subject}:{self.description}"
        return hashlib.md5(content.encode()).hexdigest()

@dataclass
class Conversation:
    """Raw conversation data to be processed."""
    id: str
    customer_id: str
    messages: List[Dict]  # List of message dicts with 'role', 'content', 'timestamp'
    metadata: Dict = field(default_factory=dict)

@dataclass
class TicketCandidate:
    """A ticket candidate extracted from conversation."""
    source_conversation_id: str
    subject: str
    description: str
    order_id: Optional[str] = None
    customer_id: Optional[str] = None
    extracted_at: datetime = field(default_factory=datetime.now)
    related_to: Optional[str] = None  # ID of related existing ticket

class IssueSplitter:
    """
    Analyzes raw conversations for multiple unrelated problems.
    Splits conversations with multiple distinct issues into separate ticket candidates.
    
    IMPROVED: Uses multi-factor detection (separators, topic shifts, problem indicators)
    instead of relying solely on separator keywords.
    """
    
    # Keywords that explicitly indicate a separate issue
    ISSUE_SEPARATOR_KEYWORDS = [
        "also", "additionally", "another issue", "separate problem",
        "different question", "unrelated", "by the way", "second issue",
        "third issue", "next problem", "apart from", "besides", "furthermore",
        "moreover", "in addition", "separately", "on another note"
    ]
    
    # Keywords that indicate a problem/issue start
    PROBLEM_INDICATORS = [
        "issue", "problem", "error", "not working", "doesn't work", "broken",
        "can't", "cannot", "won't", "unable", "failed", "failure", "crash",
        "stuck", "help", "assist", "refund", "complaint", "bug", "glitch",
        "trouble", "concern", "delay", "wrong", "incorrect", "missing"
    ]
    
    # Words that suggest continuation/elaboration (NOT topic shifts)
    CONTINUATION_KEYWORDS = [
        "for example", "specifically", "details", "details about", "more about",
        "regarding", "concerning", "related to", "about", "with", "with respect to",
        "with regard to", "as for", "thing is", "thing about"
    ]
    
    def __init__(self):
        self.separator_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.ISSUE_SEPARATOR_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.problem_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.PROBLEM_INDICATORS) + r')\b',
            re.IGNORECASE
        )
        self.continuation_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.CONTINUATION_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
    
    def split_conversation(self, conversation: Conversation) -> List[TicketCandidate]:
        """
        Analyze a conversation and split it into separate ticket candidates if needed.
        
        Returns a list of TicketCandidate objects, one per distinct issue found.
        """
        # Extract all text content from messages
        all_text = " ".join([
            msg.get("content", "") for msg in conversation.messages
            if msg.get("role") in ["user", "customer"]
        ])
        
        # IMPROVED: Use multi-factor detection instead of just separator keywords
        issue_boundaries = self._detect_issue_boundaries(all_text)
        
        if len(issue_boundaries) <= 2:
            # Single issue - create one candidate
            return [self._create_single_candidate(conversation, all_text)]
        
        # Multiple issues detected - split them
        return self._split_into_candidates(conversation, all_text, issue_boundaries)
    
    def _detect_issue_boundaries(self, text: str) -> List[int]:
        """
        IMPROVED: Detect boundaries between distinct issues using multiple factors.
        
        Returns a list of character positions where issue boundaries occur.
        Includes position 0 (start) and position len(text) (end).
        """
        boundaries = [0]
        
        # Split into sentences for analysis
        sentences = re.split(r'(?<=[.!?])\s+', text)
        char_pos = 0
        
        for i, sentence in enumerate(sentences):
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                char_pos += len(sentence) + 1
                continue
            
            # Check if this sentence marks a boundary
            if i > 0 and self._is_boundary_sentence(sentence_stripped, sentences[i-1].strip() if i > 0 else ""):
                boundaries.append(char_pos)
            
            char_pos += len(sentence) + 1
        
        boundaries.append(len(text))
        
        # Remove duplicate boundaries and sort
        boundaries = sorted(set(boundaries))
        
        # Only split if we have at least 2 distinct segments with meaningful content
        if len(boundaries) > 2:
            # Verify that segments are actually distinct (not just short separators)
            segments = [text[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries)-1)]
            distinct_segments = [s for s in segments if len(s.strip()) > 50]  # At least 50 chars
            
            if len(distinct_segments) >= 2:
                return boundaries
        
        return [0, len(text)]
    
    def _is_boundary_sentence(self, current_sentence: str, previous_sentence: str) -> bool:
        """
        IMPROVED: Determine if current sentence marks a boundary to a new issue.
        
        Returns True if sentence indicates transition to unrelated problem.
        """
        current_lower = current_sentence.lower()
        previous_lower = previous_sentence.lower()
        
        # Factor 1: Explicit separator keywords (high confidence)
        if self.separator_pattern.search(current_lower):
            # But not if it's just elaboration
            if not self.continuation_pattern.search(current_lower):
                return True
        
        # Factor 2: Multiple problem indicators in close succession (new problem starting)
        # Current sentence starts with problem indicator AND previous had different problem
        problem_count_current = len(self.problem_pattern.findall(current_lower))
        problem_count_previous = len(self.problem_pattern.findall(previous_lower))
        
        # If current sentence has problem indicator AND previous was different domain
        if problem_count_current >= 1 and problem_count_previous >= 1:
            # Check if they're discussing different domains (keywords don't overlap)
            current_keywords = set(re.findall(r'\b\w+\b', current_lower)) & set(self.PROBLEM_INDICATORS)
            previous_keywords = set(re.findall(r'\b\w+\b', previous_lower)) & set(self.PROBLEM_INDICATORS)
            
            # Different problem keywords = different issues
            if current_keywords != previous_keywords and len(current_keywords & previous_keywords) == 0:
                return True
        
        # Factor 3: Topic shift indicators (e.g., "about X" then "about Y" with unrelated content)
        if self._has_topic_shift(previous_lower, current_lower):
            return True
        
        return False
    
    def _has_topic_shift(self, previous_sentence: str, current_sentence: str) -> bool:
        """
        IMPROVED: Detect topic/domain shifts between consecutive sentences.
        
        Returns True if sentences discuss unrelated topics.
        """
        # Extract key nouns/domains from sentences
        prev_nouns = set(re.findall(r'\b([a-z]{4,})\b', previous_sentence))
        curr_nouns = set(re.findall(r'\b([a-z]{4,})\b', current_sentence))
        
        # Remove common stop words to focus on meaningful content
        common_words = {
            'this', 'that', 'have', 'from', 'with', 'when', 'what', 'your',
            'order', 'issue', 'problem', 'help', 'need', 'want', 'please'
        }
        prev_nouns -= common_words
        curr_nouns -= common_words
        
        # If very little word overlap, likely different topics
        if prev_nouns and curr_nouns:
            overlap = prev_nouns & curr_nouns
            total = prev_nouns | curr_nouns
            similarity = len(overlap) / len(total) if total else 0
            
            # Low similarity + current has problem indicator = topic shift
            if similarity < 0.3 and self.problem_pattern.search(current_sentence):
                return True
        
        return False
    
    def _create_single_candidate(self, conversation: Conversation, text: str) -> TicketCandidate:
        """Create a single ticket candidate from the conversation."""
        subject = self._extract_subject(text)
        order_id = self._extract_order_id(text)
        
        return TicketCandidate(
            source_conversation_id=conversation.id,
            customer_id=conversation.customer_id,
            subject=subject,
            description=text[:2000],
            order_id=order_id
        )
    
    def _split_into_candidates(self, conversation: Conversation, text: str, boundaries: List[int]) -> List[TicketCandidate]:
        """
        IMPROVED: Split the text into multiple ticket candidates based on detected boundaries.
        Now uses detected boundaries instead of keyword-based splitting.
        """
        candidates = []
        
        # Extract text segments between boundaries
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = text[start:end].strip()
            
            # Skip very short segments (noise)
            if len(segment) < 30:
                continue
            
            subject = self._extract_subject(segment)
            order_id = self._extract_order_id(segment)
            
            candidate = TicketCandidate(
                source_conversation_id=conversation.id,
                customer_id=conversation.customer_id,
                subject=f"{subject} (Issue {len(candidates) + 1})" if len(boundaries) > 3 else subject,
                description=segment[:2000],
                order_id=order_id
            )
            candidates.append(candidate)
        
        # If no candidates created, fall back to single candidate
        if not candidates:
            return [self._create_single_candidate(conversation, text)]
        
        return candidates
    
    def _extract_subject(self, text: str) -> str:
        """Extract a subject line from the text."""
        # Take first sentence or first 100 chars
        first_sentence = re.split(r'[.!?]', text)[0].strip()
        return first_sentence[:100] if first_sentence else "Support Request"
    
    def _extract_order_id(self, text: str) -> Optional[str]:
        """Extract order ID from the text if present."""
        # Common order ID patterns
        patterns = [
            r'order[\s#:]*([A-Z0-9]{4,})',
            r'order[\s#:]*(\d{4,})',
            r'#([A-Z0-9]{4,})',
            r'#(\d{5,})',
            r'order\s*id[\s:]*([A-Z0-9]{4,})',
            r'order\s*id[\s:]*(\d{4,})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

class DuplicateDetector:
    """
    Compares incoming requests against existing open/resolved tickets.
    Detects duplicates using semantic similarity, customer ID, order ID, and embeddings.
    """
    
    # Similarity thresholds
    SEMANTIC_SIMILARITY_THRESHOLD = 0.85
    EXACT_MATCH_THRESHOLD = 0.95
    
    def __init__(self, existing_tickets: Optional[List[Ticket]] = None):
        self.existing_tickets = existing_tickets or []
    
    def check_duplicate(self, candidate: TicketCandidate) -> Tuple[bool, Optional[str], float]:
        """
        Check if the candidate is a duplicate of an existing ticket.
        
        Returns:
            Tuple of (is_duplicate, matched_ticket_id, confidence_score)
        """
        best_match_id = None
        best_score = 0.0
        
        for ticket in self.existing_tickets:
            if ticket.status == "closed":
                continue  # Skip closed tickets for duplicate detection
            
            score = self._compute_similarity(candidate, ticket)
            if score > best_score:
                best_score = score
                best_match_id = ticket.id
        
        is_duplicate = best_score >= self.SEMANTIC_SIMILARITY_THRESHOLD
        return is_duplicate, best_match_id, best_score
    
    def _compute_similarity(self, candidate: TicketCandidate, ticket: Ticket) -> float:
        """
        Compute similarity score between a candidate and existing ticket.
        Returns a score between 0.0 and 1.0.
        """
        scores = []
        weights = []
        
        # 1. Customer ID match (medium weight)
        if candidate.customer_id and ticket.customer_id and candidate.customer_id == ticket.customer_id:
            scores.append(1.0)
            weights.append(0.25)
        else:
            scores.append(0.0)
            weights.append(0.25)
        
        # 2. Order ID match (high weight)
        if candidate.order_id and ticket.order_id and candidate.order_id == ticket.order_id:
            scores.append(1.0)
            weights.append(0.35)
        else:
            scores.append(0.0)
            weights.append(0.35)
        
        # 3. Text similarity (embeddings when available, otherwise semantic similarity)
        text_sim = self._compute_text_similarity(candidate, ticket)
        scores.append(text_sim)
        weights.append(0.40)
        
        # Weighted average
        if sum(weights) == 0:
            return 0.0
        
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    def _compute_text_similarity(self, candidate: TicketCandidate, ticket: Ticket) -> float:
        """
        Compute text similarity using embeddings if available, otherwise use improved text similarity.
        Returns a score between 0.0 and 1.0.
        """
        # Check if embeddings are available on both
        if candidate.embedding and ticket.embedding:
            return self._cosine_similarity(candidate.embedding, ticket.embedding)
        
        # Improved semantic similarity using word overlap + structural similarity
        candidate_text = f"{candidate.subject} {candidate.description}".lower().strip()
        ticket_text = f"{ticket.subject} {ticket.description}".lower().strip()
        
        # Exact match check
        if candidate_text == ticket_text:
            return 1.0
        
        # Extract words (improved from pure n-gram approach)
        candidate_words = set(re.findall(r'\b\w+\b', candidate_text))
        ticket_words = set(re.findall(r'\b\w+\b', ticket_text))
        
        # Remove common stop words to focus on meaningful content
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her'
        }
        candidate_words -= stop_words
        ticket_words -= stop_words
        
        if not candidate_words or not ticket_words:
            # Fall back to character n-gram if no meaningful words
            return self._ngram_similarity(candidate_text, ticket_text)
        
        # Jaccard similarity on meaningful words
        intersection = candidate_words & ticket_words
        union = candidate_words | ticket_words
        word_sim = len(intersection) / len(union) if union else 0.0
        
        # Character n-gram similarity for structural similarity
        ngram_sim = self._ngram_similarity(candidate_text, ticket_text)
        
        # Combine word and structural similarity
        return 0.7 * word_sim + 0.3 * ngram_sim
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        
        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _ngram_similarity(self, text1: str, text2: str) -> float:
        """
        Compute character n-gram similarity (fallback/supplementary).
        """
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union) if union else 0.0

class IssueGrouper:
    """
    Links or merges related issues belonging to the same root incident or customer thread.
    """
    
    def __init__(self, existing_tickets: Optional[List[Ticket]] = None):
        self.existing_tickets = existing_tickets or []
        self.groups: Dict[str, List[str]] = {}  # group_id -> list of ticket_ids
    
    def group_related_issues(self, candidate: TicketCandidate) -> Tuple[str, List[str]]:
        """
        Analyze a ticket candidate and determine if it belongs to an existing group.
        
        Returns:
            Tuple of (group_id, list_of_related_ticket_ids_in_group)
        """
        related_tickets = []
        
        # Check for customer thread grouping
        for ticket in self.existing_tickets:
            if self._belongs_to_same_thread(candidate, ticket):
                related_tickets.append(ticket.id)
        
        # Check for same root incident
        root_cause_group = self._find_root_cause_group(candidate)
        if root_cause_group:
            related_tickets.extend(self.groups.get(root_cause_group, []))
        
        # Deduplicate
        related_tickets = list(set(related_tickets))
        
        if related_tickets:
            group_id = f"group_{candidate.source_conversation_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.groups[group_id] = related_tickets
            return group_id, related_tickets
        
        return "", []
    
    def _belongs_to_same_thread(self, candidate: TicketCandidate, ticket: Ticket) -> bool:
        """
        Check if candidate belongs to the same customer thread as existing ticket.
        """
        # Same customer
        if not candidate.customer_id or candidate.customer_id != ticket.customer_id:
            return False
        
        # Same order context
        if candidate.order_id and ticket.order_id and candidate.order_id == ticket.order_id:
            return True
        
        # Same subject matter (simple keyword matching)
        candidate_keywords = set(candidate.subject.lower().split())
        ticket_keywords = set(ticket.subject.lower().split())
        common_keywords = candidate_keywords & ticket_keywords
        
        # If significant keyword overlap and same customer, likely related
        if len(common_keywords) >= 2:
            return True
        
        return False
    
    def _find_root_cause_group(self, candidate: TicketCandidate) -> Optional[str]:
        """Find if candidate belongs to a root cause group."""
        # Extract potential root cause indicators from description
        root_indicators = self._extract_root_indicators(candidate.description)
        
        for group_id, ticket_ids in self.groups.items():
            # Check if any ticket in group shares root indicators
            for ticket in self.existing_tickets:
                if ticket.id in ticket_ids:
                    ticket_indicators = self._extract_root_indicators(ticket.description)
                    if root_indicators & ticket_indicators:
                        return group_id
        
        return None
    
    def _extract_root_indicators(self, text: str) -> Set[str]:
        """Extract potential root cause indicators from text."""
        indicators = set()
        
        # Look for error codes, system names, common issue patterns
        patterns = [
            r'error\s*(?:code)?\s*[:#]?\s*(\w+)',
            r'system\s*[:#]?\s*(\w+)',
            r'\b([A-Z]{3,10})\b',  # Acronyms
            r'\b(\d{4,})\b',  # Long numbers (order IDs, etc)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.update(matches)
        
        return indicators

class IngestionProcessor:
    """
    Main processor that orchestrates Step 1 of Task 3:
    - Issue Splitting
    - Duplicate Detection
    - Issue Grouping
    """
    
    def __init__(self, existing_tickets: Optional[List[Ticket]] = None):
        self.existing_tickets = existing_tickets or []
        self.splitter = IssueSplitter()
        self.duplicate_detector = DuplicateDetector(self.existing_tickets)
        self.grouper = IssueGrouper(self.existing_tickets)
    
    def process(self, conversation: Conversation) -> Dict:
        """
        Process a conversation through Step 1 pipeline.
        
        Returns a dict with:
        - candidates: List of ticket candidates
        - duplicates: List of duplicate detections
        - groups: List of group assignments
        """
        results = {
            "candidates": [],
            "duplicates": [],
            "groups": [],
            "conversation_id": conversation.id
        }
        
        # Step 1a: Issue Splitting
        candidates = self.splitter.split_conversation(conversation)
        results["candidates"] = candidates
        
        # Step 1b: Duplicate Detection and 1c: Issue Grouping
        for candidate in candidates:
            # Check for duplicates
            is_duplicate, match_id, score = self.duplicate_detector.check_duplicate(candidate)
            if is_duplicate:
                results["duplicates"].append({
                    "candidate": candidate,
                    "matched_ticket_id": match_id,
                    "confidence": score
                })
            
            # Check for related issues / grouping
            group_id, related_tickets = self.grouper.group_related_issues(candidate)
            if group_id:
                results["groups"].append({
                    "candidate": candidate,
                    "group_id": group_id,
                    "related_ticket_ids": related_tickets
                })
        
        return results
