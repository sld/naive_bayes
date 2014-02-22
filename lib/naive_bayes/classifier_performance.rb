module NaiveBayes
  module ClassifierPerformance

    def precision( confusion_matrix, klass )
      confusion_matrix[klass][klass].to_i / confusion_matrix.values.inject(0.0){ |s,e| s += e[klass].to_f }
    end


    def recall(confusion_matrix, klass)
      confusion_matrix[klass][klass].to_f / confusion_matrix[klass].values.inject(0){|s,e| s+=e}
    end


    def accuracy( confusion_matrix )
      val = 0.0
      denom = 0.0 # Count of all documents in test
      klasses = confusion_matrix.keys
      klasses.each do |klass|
        denom += confusion_matrix.values.inject(0.0){ |s,e| s += e[klass].to_f }
      end
      klasses.each do |klass|
        val += confusion_matrix[klass][klass].to_i / denom
      end
      val
    end


    def f_measure(confusion_matrix, klass, beta=1)
      precision = precision(confusion_matrix, klass)
      recall = recall(confusion_matrix, klass)
      ( (beta**2 + 1) * precision * recall )/( beta**2 * precision + recall )
    end

  end
end