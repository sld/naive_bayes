require "naive_bayes/version"
# require "naive_bayes/classifier_performance"
require 'set'

module NaiveBayes


  class NaiveBayes
    ROSE = :rose
    MULTINOMIAL = :multinomial

    # By default each part of string divided by space(" ") is feature
    # Type can be: +:multinomial+ OR +:rose+   
    # If type is rose then you need to specify [:rose][:duplicate_count] and [:rose][:duplicate_klass] from options 
    def initialize laplace_smoothing = 1.0, type = :multinomial, options = {}
      # Class' documents count 
      # Example { :japanese => 3 } means that 3 documents with class :japanese
      @klass_docs_count = {}     
      # Class' words count  
      # Example: { :japanese => {"Tokyo" => 3} } means that in class :japanese, word "Tokyo" was 3 times
      @klass_words_count = {}
      @vocabulary = Set.new
      @laplace_smoothing = laplace_smoothing

      @type = type
      if @type == ROSE 
        raise ArgumentError if !options[:rose][:duplicate_klass] || !options[:rose][:duplicate_count]
        # Example {:japanese => [3,4,5]}
        @average_document_words = {}
        # Example {:japanese => 10}
        @rose_duplicate_count = {}
        @m_rose = {}
        @rose_duplicate_count[options[:rose][:duplicate_klass]] = options[:rose][:duplicate_count]
      end
    end


    def train( string, klass )  
      tokens = form_features_vector( string )           
      @klass_words_count[klass] ||= {}      
      tokens.each do |token|    
        @vocabulary << token
        @klass_words_count[klass][token] = @klass_words_count[klass][token].to_i + 1 
      end
      @klass_docs_count[klass] = @klass_docs_count[klass].to_i + 1    

      if @type == ROSE
        @average_document_words[klass] ||= []
        @average_document_words[klass] << tokens.count  
      end
    end


    def classify( string )
      raise 'Run "precache!" function before classifing!' unless @precached

      klasses = @klass_docs_count.keys
      klass_probs = {}
      features_vector = form_features_vector( string )  
      
      klasses.each do |klass| 
        case @type 
        when MULTINOMIAL 
          klass_probs[klass] = document_class_prob( features_vector, klass ) 
        when ROSE
          klass_probs[klass] = rose_document_class_prob( features_vector, klass ) 
        end
      end

      get_necessary_klass( klass_probs )     
    end


    def document_class_prob( features_vector, klass )
      product_of_cond_probs = 1
      features_vector.each{ |e| product_of_cond_probs *= cond_prob( e, klass ) }
      class_prob( klass ) * product_of_cond_probs
    end


    # Logarithmic version to avoid Arithmetic_underflow
    #NOTE: not used currently
    def log_document_class_prob( features_vector, klass )
      log_sum_of_cond_probs = 0
      features_vector.each{ |e| log_sum_of_cond_probs += Math::log( cond_prob( e, klass ) ) } 
      log_val = Math::log( class_prob( klass ) ) + log_sum_of_cond_probs    
    end


    #---------------------------- ROSE --------------------------------------------
    # Implementation NB using ROSE smoothing. Described in paper "Smoothing Multinomial Naïve Bayes in the Presence of Imbalance"
    def rose_document_class_prob( features_vector, klass )
      product_of_cond_probs = features_vector.inject(1){ |product, e| product *= rose_cond_prob( e, klass ) }
      class_prob( klass ) * product_of_cond_probs
    end


    def rose_cond_prob token, klass 
      all_words_in_klass = @all_words_in_klass_cached[klass]     
      sum_of_m_rose = @sum_of_m_rose_cached[klass]
      ( @klass_words_count[klass][token].to_i + @laplace_smoothing + m_rose(token, klass) ) / ( all_words_in_klass + @laplace_smoothing * @vocabulary.count + sum_of_m_rose )
    end


    def m_rose(token, klass)
      @m_rose[klass] ||= {}
      if @m_rose[klass][token]        
        return @m_rose[klass][token]
      else
        all_words_in_klass = @all_words_in_klass_cached[klass]
        @m_rose[klass][token] = (rose_duplicate_count(klass) * average_document_words( klass ) * @klass_words_count[klass][token].to_f) / all_words_in_klass        
        return @m_rose[klass][token]
      end
    end


    def average_document_words( klass )     
      @mean ||= {}
      if @mean[klass]
        return @mean[klass]
      else
        mean = @average_document_words[klass].reduce(:+) / @average_document_words[klass].count.to_f
        @mean[klass] = mean.round
        return @mean[klass]
      end
    end


    def rose_duplicate_count( klass )  
      @rose_duplicate_count[klass].to_i                                              
    end
    #-----END----------------------- ROSE ------------------------------------END--------


    def cond_prob( token, klass )   
      @cond_prob_cached ||= {}
      @cond_prob_cached[klass] ||= {}
      return @cond_prob_cached[klass][token] if @cond_prob_cached[klass][token]
      all_words_in_klass = @all_words_in_klass_cached[klass]
      @cond_prob_cached[klass][token] = ( @klass_words_count[klass][token].to_i + @laplace_smoothing ) / ( all_words_in_klass + @laplace_smoothing * @vocabulary.count )
    end


    def class_prob( klass )
      @cached_klass_prob ||= {}
      return @cached_klass_prob[klass] if @cached_klass_prob[klass]
      @cached_klass_prob[klass] = @klass_docs_count[klass].to_f / @klass_docs_count.values.inject{ |e,s| s += e }
    end


    def export
      export_hash = { :docs_count => @klass_docs_count,
                      :words_count => @klass_words_count,
                      :vocabulary => @vocabulary,
                      :laplace_smoothing => @laplace_smoothing }
      export_hash.merge!({:rose_duplicate_count => @rose_duplicate_count, :average_document_words => @average_document_words}) if @type == ROSE
      return export_hash
    end


    def import!( klass_docs_count, klass_words_count, vocabulary, options={} )
      @klass_docs_count = klass_docs_count
      @klass_words_count = klass_words_count
      @vocabulary = vocabulary
      @laplace_smoothing = options[:laplace_smoothing] if options[:laplace_smoothing]
      precache_all_words_in_class

      if @type == ROSE
        @rose_duplicate_count = options[:rose_duplicate_count] if options[:rose_duplicate_count]
        @average_document_words = options[:average_document_words] if options[:average_document_words]
      end
    end


    def form_features_vector obj 
      case obj
        when Array
          return obj
        when String 
          return get_features( obj )
      end     
    end


    def precache!
      @precached = true
      precache_all_words_in_class
      precache_sum_of_m_rose if @type == ROSE
    end


    protected


    def get_features( string )
      string.split(" ")
    end


    def get_necessary_klass( klass_probs )
      klass_probs = Hash[klass_probs.sort_by{|k,v| v}.reverse]
      max_klass_prob = klass_probs.first
      {:class => max_klass_prob[0], :value => max_klass_prob[1], :all_values => klass_probs.values}
    end


    def precache_all_words_in_class
      @all_words_in_klass_cached = {}
      @klass_docs_count.keys.each do |klass|
        @all_words_in_klass_cached[klass] = @klass_words_count[klass].values.inject{ |e,s| s = s + e }.to_f         
      end
    end


    def precache_sum_of_m_rose
      @sum_of_m_rose_cached = {}
      @klass_docs_count.keys.each do |klass|
        sum_of_m_rose = 0.0
        @klass_words_count[klass].keys.each{ |tkn| sum_of_m_rose += m_rose(tkn, klass) }
        @sum_of_m_rose_cached[klass] = sum_of_m_rose
      end
    end

  end
end
