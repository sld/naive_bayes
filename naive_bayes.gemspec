# -*- encoding: utf-8 -*-
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'naive_bayes/version'

Gem::Specification.new do |gem|
  gem.name          = "naive_bayes"
  gem.version       = NaiveBayes::VERSION
  gem.authors       = ["Idris"]
  gem.email         = ["sld7700@gmail.com"]
  gem.description   = %q{Naive Bayes implementation}
  gem.summary       = %q{Simple Naive Bayes implementation}
  gem.homepage      = ""

  gem.files         = `git ls-files`.split($/)
  gem.executables   = gem.files.grep(%r{^bin/}).map{ |f| File.basename(f) }
  gem.test_files    = gem.files.grep(%r{^(test|spec|features)/})
  gem.require_paths = ["lib"]

  gem.add_development_dependency "rspec"
end
